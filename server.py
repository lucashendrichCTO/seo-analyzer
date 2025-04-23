from waitress import serve
from app import app

if __name__ == '__main__':
    print("Starting SEO Analyzer on http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080)
