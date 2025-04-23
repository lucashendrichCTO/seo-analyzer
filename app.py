from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter
import re
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import pandas as pd
from nltk.util import ngrams
import time
import traceback
import threading
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

# Cache for sentence and word tokenizers
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.WordPunctTokenizer()

# Common words to exclude from keyword analysis
EXCLUDED_KEYWORDS = {'that'}.union(set(nltk.corpus.stopwords.words('english')))

# SEO benchmarks for different content types
SEO_BENCHMARKS = {
    'article': {
        'primary_keyword_density': (1.5, 2.5),  # (min%, max%)
        'secondary_keyword_density': (0.5, 1.5),
        'min_word_count': 300,
        'optimal_word_count': 1000,
    },
    'blog_post': {
        'primary_keyword_density': (1.0, 2.0),
        'secondary_keyword_density': (0.5, 1.0),
        'min_word_count': 500,
        'optimal_word_count': 1500,
    },
    'landing_page': {
        'primary_keyword_density': (2.0, 3.0),
        'secondary_keyword_density': (1.0, 2.0),
        'min_word_count': 200,
        'optimal_word_count': 800,
    }
}

def extract_key_phrases(text):
    """Extract key phrases including multi-word terms from text."""
    # Use cached tokenizers for better performance
    sentences = sentence_tokenizer.tokenize(text)
    all_phrases = []
    
    # Pre-compile regex pattern for performance
    alnum_pattern = re.compile(r'[a-zA-Z0-9]')
    
    # Process sentences in chunks for better performance
    for sentence in sentences:
        # Use cached word tokenizer
        words = word_tokenizer.tokenize(sentence)
        tagged = pos_tag(words)
        
        # Use set for faster lookups
        valid_words = {
            word for word, tag in tagged 
            if tag.startswith('NN') 
            and word.lower() not in EXCLUDED_KEYWORDS 
            and alnum_pattern.search(word)
        }
        
        # Add single words
        all_phrases.extend(valid_words)
        
        # Generate n-grams only from valid words for better performance
        if len(valid_words) > 1:
            word_list = list(valid_words)
            all_phrases.extend(' '.join(gram) for gram in ngrams(word_list, 2))
            if len(valid_words) > 2:
                all_phrases.extend(' '.join(gram) for gram in ngrams(word_list, 3))
    
    return all_phrases

# Cache for Google Trends data
trend_cache = {}
trend_cache_lock = threading.Lock()
TREND_CACHE_DURATION = 3600  # Cache for 1 hour

def get_trend_data(keywords):
    """Get trend data for keywords using Google Trends with caching."""
    def get_default_trend():
        return {
            'average_interest': 50,
            'recent_interest': 50,
            'trending_up': False
        }

    if not keywords:
        return {}

    trend_scores = {}
    current_time = time.time()

    # First check cache for all keywords
    for keyword in keywords[:5]:
        with trend_cache_lock:
            if keyword in trend_cache:
                cached_data, cache_time = trend_cache[keyword]
                if current_time - cache_time < TREND_CACHE_DURATION:
                    trend_scores[keyword] = cached_data
                    continue

        # If not in cache, initialize with default values
        trend_scores[keyword] = get_default_trend()

    try:
        # Initialize pytrends
        print("\n=== Initializing Google Trends ===")
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
        
        # Process uncached keywords with significant delays
        base_delay = 5.0  # Start with 5 second delay
        max_retries = 3
        
        for keyword in [k for k in keywords[:5] if trend_scores[k] == get_default_trend()]:
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Normalize keyword
                    norm_keyword = normalize_keyword(keyword)
                    if not norm_keyword or len(norm_keyword.strip()) < 2:
                        print(f"Skipping invalid keyword: '{keyword}' (normalized: '{norm_keyword}')")
                        break

                    print(f"\nFetching trend data for: '{norm_keyword}'")
                    
                    # Add significant delay between requests
                    current_delay = base_delay * (2 ** retry_count)
                    print(f"Waiting {current_delay} seconds before request...")
                    time.sleep(current_delay)
                    
                    # Get trend data
                    pytrends.build_payload([norm_keyword], timeframe='today 3-m')
                    data = pytrends.interest_over_time()

                    if data.empty:
                        print(f"No trend data returned for '{norm_keyword}'")
                        break

                    if norm_keyword not in data.columns:
                        print(f"Keyword '{norm_keyword}' not found in response columns: {data.columns}")
                        break

                    values = data[norm_keyword].values
                    avg = float(values.mean())
                    recent = float(values[-7:].mean() if len(values) >= 7 else values.mean())

                    print(f"Trend data for '{norm_keyword}':")
                    print(f"- Average interest: {avg:.1f}")
                    print(f"- Recent interest: {recent:.1f}")
                    print(f"- Trending up: {recent > avg}")

                    result = {
                        'average_interest': round(avg, 1),
                        'recent_interest': round(recent, 1),
                        'trending_up': recent > avg
                    }

                    # Update both cache and results
                    with trend_cache_lock:
                        trend_cache[keyword] = (result, current_time)
                    trend_scores[keyword] = result
                    success = True

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if "429" in str(e):
                        if retry_count < max_retries:
                            wait_time = 10 * (2 ** retry_count)  # 20s, 40s, 80s
                            print(f"Rate limit hit for '{keyword}', waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Max retries reached for '{keyword}', using default values")
                    else:
                        print(f"Request error for '{keyword}': {str(e)}")
                        break
                        
                except Exception as e:
                    print(f"Error processing trend data for '{keyword}':")
                    print(f"- Error type: {type(e).__name__}")
                    print(f"- Error message: {str(e)}")
                    traceback.print_exc()
                    break

    except Exception as e:
        print("\n=== Fatal error in trend analysis ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()

    print("\n=== Final trend scores ===")
    for kw, score in trend_scores.items():
        print(f"{kw}: {score}")

    return trend_scores

def normalize_keyword(keyword):
    """Normalize a keyword for Google Trends."""
    # Convert to lowercase
    keyword = keyword.lower().strip()
    
    # Remove special characters but keep spaces
    keyword = re.sub(r'[^a-z0-9\s]', '', keyword)
    
    # Replace multiple spaces with single space
    keyword = re.sub(r'\s+', ' ', keyword)
    
    return keyword.strip()

def calculate_keyword_metrics(word_count, keyword_freq):
    """Calculate keyword metrics including density and trend data."""
    try:
        # Convert keyword_freq to a list if it's a dict_items object
        if not isinstance(keyword_freq, list):
            keyword_freq = list(keyword_freq)
            
        # Get trend data for top keywords
        trend_data = get_trend_data([kw for kw, _ in keyword_freq])
        
        # Calculate metrics
        metrics = []
        for keyword, count in keyword_freq:
            frequency = round((count / word_count) * 100, 0)  # Round to nearest whole number
            metric = {
                'keyword': keyword,
                'count': count,  # Include the raw count
                'frequency': frequency
            }
            
            # Add trend data if available
            if keyword in trend_data:
                metric.update(trend_data[keyword])
            else:
                metric.update({
                    'average_interest': 50,  # Default value
                    'recent_interest': 50,   # Default value
                    'trending_up': False
                })
            
            metrics.append(metric)
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        # Return basic metrics without trend data
        return [{
            'keyword': kw,
            'count': count,
            'frequency': round((count / word_count) * 100, 0),
            'average_interest': 50,
            'recent_interest': 50,
            'trending_up': False
        } for kw, count in keyword_freq]

def suggest_meta_tags(content, key_phrases):
    meta_suggestions = {
        'title': '',
        'description': '',
        'keywords': []
    }
    
    # Get the first sentence for description
    first_sentence = sent_tokenize(content)[:1]
    if first_sentence:
        meta_suggestions['description'] = first_sentence[0][:160]
    
    # Use top key phrases for title and keywords
    top_phrases = Counter(key_phrases).most_common(5)
    if top_phrases:
        meta_suggestions['title'] = f"{top_phrases[0][0].title()} - {' '.join(p[0].title() for p in top_phrases[1:3])}"
        meta_suggestions['keywords'] = [p[0] for p in top_phrases]
    
    return meta_suggestions

def analyze_seo(content, content_type='article'):
    results = {
        'word_count': 0,
        'keyword_metrics': [],
        'meta_tags': {},
        'headings': {
            'h1': 0, 'h2': 0, 'h3': 0
        },
        'suggestions': [],
        'suggested_meta_tags': {},
        'suggested_keywords': [],
        'llm_readability_score': 0,
        'llm_readability_explanation': '',
        'llm_readability_suggestions': [],
        'llm_attention_scores': {},
        'llm_embedding_scores': {}
    }
    
    try:
        # Parse content
        soup = BeautifulSoup(content, 'html.parser') if content_type == 'html' else None
        text_content = soup.get_text() if soup else content
        
        # Calculate word count properly
        words = word_tokenize(text_content)
        word_count = len([w for w in words if any(c.isalnum() for c in w)])
        results['word_count'] = word_count
        
        # Extract key phrases (including multi-word terms)
        key_phrases = extract_key_phrases(text_content)
        
        # Count frequencies of both single words and phrases
        phrase_freq = Counter(key_phrases)
        
        # Sort by frequency and get top phrases
        top_phrases = phrase_freq.most_common(10)
        
        # Filter out single words if their bigram exists with higher frequency
        filtered_phrases = []
        skip_words = set()
        
        for phrase, freq in top_phrases:
            words = phrase.split()
            if len(words) > 1:
                # Add multi-word phrase and mark its components to be skipped
                filtered_phrases.append((phrase, freq))
                skip_words.update(words)
            elif phrase not in skip_words:
                # Add single word if it's not part of a more frequent multi-word phrase
                filtered_phrases.append((phrase, freq))
        
        # Take top 5 filtered phrases
        top_keywords = filtered_phrases[:5]
        
        print("\n=== Top Keywords for Analysis ===")
        print(f"Found {len(top_keywords)} keywords: {[kw for kw, _ in top_keywords]}")
        
        # Calculate keyword metrics including trend data
        results['keyword_metrics'] = calculate_keyword_metrics(word_count, top_keywords)
        
        # Analyze meta tags
        meta_tags = soup.find_all('meta') if soup else []
        for tag in meta_tags:
            if tag.get('name'):
                results['meta_tags'][tag.get('name')] = tag.get('content')
        
        # Count headings
        for h_level in ['h1', 'h2', 'h3']:
            results['headings'][h_level] = len(soup.find_all(h_level)) if soup else 0
        
        # Generate meta tag suggestions
        results['suggested_meta_tags'] = suggest_meta_tags(text_content, key_phrases)
        
        # Generate keyword suggestions
        existing_keywords = set(item['keyword'].lower() for item in results['keyword_metrics'])
        suggested_keywords = set(p.lower() for p in key_phrases if len(p.split()) > 1)  # Only use multi-word phrases
        results['suggested_keywords'] = list(suggested_keywords - existing_keywords)[:5]
        
        # Add word count suggestions
        benchmarks = SEO_BENCHMARKS[content_type]
        if len(word_tokenize(text_content)) < benchmarks['min_word_count']:
            results['suggestions'].append(
                f"Content length ({len(word_tokenize(text_content))} words) is below recommended minimum of {benchmarks['min_word_count']} words"
            )
        elif len(word_tokenize(text_content)) < benchmarks['optimal_word_count']:
            results['suggestions'].append(
                f"Consider expanding content to reach optimal length of {benchmarks['optimal_word_count']} words"
            )
        
        # Get LLM attention scores
        results['llm_attention_scores'] = compute_llm_attention_scores(text_content)
        
        # Get embedding similarity scores for key phrases
        results['llm_embedding_scores'] = compute_embedding_similarity(text_content, [phrase for phrase, _ in top_keywords])
        
        # Generate suggestions
        if not results['meta_tags'].get('description'):
            results['suggestions'].append('Add a meta description tag')
            results['suggestions'].append(f'Suggested description: {results["suggested_meta_tags"]["description"]}')
        if not results['meta_tags'].get('keywords'):
            results['suggestions'].append('Add meta keywords tag')
            results['suggestions'].append(f'Suggested keywords: {", ".join(results["suggested_meta_tags"]["keywords"])}')
        if results['headings']['h1'] == 0:
            results['suggestions'].append('Add an H1 heading')
            
        # Add LLM-based suggestions
        if results['llm_attention_scores']:
            top_attention_words = sorted(results['llm_attention_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
            results['suggestions'].append(f"High-attention words: {', '.join(word for word, _ in top_attention_words)}")
            
        if results['llm_embedding_scores']:
            semantic_matches = sorted(results['llm_embedding_scores'].items(), key=lambda x: x[1], reverse=True)
            results['suggestions'].append(f"Most semantically relevant keyword: {semantic_matches[0][0]}")
            if semantic_matches[-1][1] < 0.5:  # If lowest similarity is < 0.5
                results['suggestions'].append(f"Consider revising keyword '{semantic_matches[-1][0]}' - low semantic relevance")
        
        # --- LLM Readability Score ---
        def compute_llm_readability_score(text, headings):
            # Heuristic: based on heading structure, avg sentence length, paragraph length, and presence of lists
            sentences = sent_tokenize(text)
            num_sentences = len(sentences)
            words = word_tokenize(text)
            num_words = len([w for w in words if any(c.isalnum() for c in w)])
            avg_sentence_length = num_words / num_sentences if num_sentences else 0
            num_paragraphs = text.count('\n\n') + 1
            h1, h2, h3 = headings['h1'], headings['h2'], headings['h3']
            # Score heuristics
            score = 1
            explanation = []
            suggestions = []
            if h1 >= 1 and h2 >= 1:
                score += 1
                explanation.append('Good use of H1/H2 headings.')
            else:
                suggestions.append('Add clear H1 and H2 headings for better structure.')
            if avg_sentence_length <= 22:
                score += 1
                explanation.append('Average sentence length is easy to read.')
            elif avg_sentence_length <= 28:
                score += 0.5
                explanation.append('Average sentence length is acceptable.')
                suggestions.append('Try to keep sentences under 22 words for better readability.')
            else:
                suggestions.append('Shorten sentences to improve readability (aim for under 22 words).')
            if num_paragraphs >= 3:
                score += 1
                explanation.append('Content is divided into multiple paragraphs.')
            else:
                suggestions.append('Break content into more paragraphs.')
            if text.count('<ul>') + text.count('<ol>') + text.count('- ') > 0:
                score += 0.5
                explanation.append('Lists/bullets detected.')
            else:
                suggestions.append('Use lists or bullet points to improve scannability.')
            score = min(round(score), 5)
            if score <= 2:
                explanation.append('Consider adding headings, shorter sentences, and paragraphs for better readability.')
            results['llm_readability_score'] = score
            results['llm_readability_explanation'] = ' '.join(explanation)
            results['llm_readability_suggestions'] = suggestions
        compute_llm_readability_score(content, results['headings'])
    
    except Exception as e:
        return {
            'error': str(e),
            'word_count': 0,
            'keyword_metrics': [],
            'meta_tags': {},
            'headings': {'h1': 0, 'h2': 0, 'h3': 0},
            'suggestions': [f"Error analyzing content: {str(e)}"],
            'suggested_meta_tags': {},
            'suggested_keywords': [],
            'llm_readability_score': 0,
            'llm_readability_explanation': '',
            'llm_readability_suggestions': [],
            'llm_attention_scores': {},
            'llm_embedding_scores': {}
        }
    
    return results

# Initialize transformer model and tokenizer lazily to save memory
model = None
tokenizer = None

def get_llm_model():
    """Lazily initialize the LLM model and tokenizer."""
    global model, tokenizer
    if model is None or tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')
            return model, tokenizer
        except Exception as e:
            print(f"Error initializing LLM model: {str(e)}")
            return None, None
    return model, tokenizer

def compute_llm_attention_scores(text, max_length=512):
    """Compute attention-based importance scores for text segments."""
    try:
        model, tokenizer = get_llm_model()
        if model is None or tokenizer is None:
            print("LLM model not available, skipping attention scores")
            return {}
            
        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
        
        # Get model outputs with attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Get attention weights from last layer
        attention = outputs.attentions[-1].mean(dim=(0, 1))  # Average over heads and batch
        
        # Convert tokens to words and aggregate attention scores
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        word_scores = {}
        current_word = ''
        current_score = 0
        
        for token, score in zip(tokens, attention):
            if token.startswith('##'):
                current_word += token[2:]
                current_score += float(score)
            else:
                if current_word:
                    word_scores[current_word] = current_score
                current_word = token
                current_score = float(score)
        
        if current_word:
            word_scores[current_word] = current_score
            
        # Normalize scores
        max_score = max(word_scores.values())
        normalized_scores = {word: score/max_score for word, score in word_scores.items()}
        
        return normalized_scores
        
    except Exception as e:
        print(f"Error in compute_llm_attention_scores: {str(e)}")
        return {}

def compute_embedding_similarity(text, keywords):
    """Compute similarity between text and keywords in embedding space."""
    try:
        model, tokenizer = get_llm_model()
        if model is None or tokenizer is None:
            print("LLM model not available, skipping embedding similarity")
            return {}
            
        # Get embeddings for text and keywords
        with torch.no_grad():
            text_inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
            text_embedding = model(**text_inputs).last_hidden_state.mean(dim=1)
            
            keyword_scores = {}
            for keyword in keywords:
                keyword_inputs = tokenizer(keyword, return_tensors='pt')
                keyword_embedding = model(**keyword_inputs).last_hidden_state.mean(dim=1)
                
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(text_embedding, keyword_embedding)
                keyword_scores[keyword] = float(similarity)
                
        return keyword_scores
        
    except Exception as e:
        print(f"Error in compute_embedding_similarity: {str(e)}")
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("\n=== Starting SEO Analysis ===")  # Debug print
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'No content provided'}), 400
            
        content = data['content']
        content_type = data.get('content_type', 'article')
        
        if not content.strip():
            return jsonify({'error': 'Content is empty'}), 400
            
        results = analyze_seo(content, content_type)
        print("\n=== Analysis Complete ===")  # Debug print
        return jsonify(results)
        
    except Exception as e:
        print(f"\n=== Error in analyze route: {str(e)} ===")
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.debug = True  # Enable debug mode
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # Force unbuffered output
    app.run(host='127.0.0.1', port=5000)
