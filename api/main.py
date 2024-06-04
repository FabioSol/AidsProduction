from flask import Flask, request, jsonify
import subprocess
from app.AidsModel import model_repo_path
from app.src.update_subrepo_data import update_subrepo_data
from app.src.predict import predict

REPO_URL = "https://github.com/FabioSol/AidsModel.git"
CLONE_DIR = "app/AidsModel"

app = Flask(__name__)




@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if not data:
        return jsonify({'message': 'No data received'}), 400

    event_type = request.headers.get('X-GitHub-Event')
    ref = data.get('ref')
    commit_message = data.get('head_commit', {}).get('message', '')

    # Check for specific commit message to prevent recursion
    if event_type == 'push' and ref == 'refs/heads/main' and '[skip ci]' not in commit_message:
        # Pull the latest changes from the subrepo
        subprocess.call(['git', 'pull', 'origin', 'main'], cwd=model_repo_path)
    return jsonify({'message': 'Action triggered'}), 200


@app.route('/update_repo_data', methods=['POST'])
def update_repo_data():
    if request.method != 'POST':
        return jsonify({'message': 'Only POST requests allowed'}), 405

    update_subrepo_data()

    # Consider adding specific paths for Git add instead of '.'
    add_command = "git add data/"  # Adjust path based on your data location
    try:
        subprocess.call(add_command.split(), cwd=model_repo_path)
    except subprocess.CalledProcessError as e:
        print(f"Error adding files: {e}")
        return jsonify({'message': 'Error updating data'}), 500

    # Allow customizing skip CI flag (optional)
    skip_ci_flag = '-m "[skip ci]"'  # Default flag
    # ... (logic to potentially modify skip_ci_flag based on request data or environment)

    commit_command = f"git commit {skip_ci_flag} -m 'Data update'"
    try:
        subprocess.call(commit_command.split(), cwd=model_repo_path)
    except subprocess.CalledProcessError as e:
        print(f"Error committing changes: {e}")
        return jsonify({'message': 'Error updating data'}), 500

    try:
        subprocess.call(['git', 'push', 'origin', 'main'], cwd=model_repo_path)
    except subprocess.CalledProcessError as e:
        print(f"Error pushing changes: {e}")
        return jsonify({'message': 'Error updating data'}), 500

    return jsonify({'message': 'Data updated and pushed (if conditions met)'}), 200

@app.route('/inference', methods=['POST'])
def inference():
    if request.method != 'POST':
        return jsonify({'message': 'Only POST requests allowed'}), 405

    data = request.json
    if not data:
        return jsonify({'message': 'No data received'}), 400

    # Perform inference using your model
    prediction = predict(data)

    return jsonify({'prediction': prediction}), 200


if __name__ == '__main__':
    app.run(port=8080)
