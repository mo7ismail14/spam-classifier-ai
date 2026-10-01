import os
from flask import Flask, jsonify
from flask_cors import CORS

from routes.spam_classifier_routes import spam_classifier_routes, load_sample_emails, list_available_spam_models
from routes.loan_approval_routes import loan_approval_routes

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['MODEL_DIR'] = os.path.join(BASE_DIR, 'model')

app.register_blueprint(spam_classifier_routes)
app.register_blueprint(loan_approval_routes)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


with app.app_context():
    load_sample_emails()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask AI Server on port {port}...")
    with app.app_context():
        print(f"Available spam models: {list_available_spam_models()}")
    app.run(host='0.0.0.0', port=port, debug=False)
