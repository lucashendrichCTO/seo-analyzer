from pytrends.request import TrendReq
import time

def test_trends():
    print("Testing Google Trends API...")
    
    try:
        # Initialize pytrends
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), retries=2, backoff_factor=0.1)
        print("Successfully initialized TrendReq")
        
        # Test with known popular terms
        keywords = ["python programming", "artificial intelligence", "machine learning"]
        success = False
        
        for keyword in keywords:
            print(f"\nTesting with keyword: {keyword}")
            
            try:
                # Build payload
                pytrends.build_payload([keyword], timeframe='today 3-m')
                print("Successfully built payload")
                
                # Get interest over time
                data = pytrends.interest_over_time()
                print(f"Data received: {not data.empty}")
                
                if not data.empty and keyword in data.columns:
                    values = data[keyword].values
                    if len(values) > 0:
                        avg = float(values.mean())
                        recent = float(values[-7:].mean() if len(values) >= 7 else values.mean())
                        print(f"Average interest: {avg:.1f}")
                        print(f"Recent interest: {recent:.1f}")
                        print(f"Trending: {'Up' if recent > avg else 'Down'}")
                        success = True
                    else:
                        print("No values found in data")
                else:
                    print("No data found for keyword")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error testing keyword {keyword}: {str(e)}")
        
        return success
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_trends()
    print(f"\nTest {'succeeded' if success else 'failed'}")
