from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize

app = Flask(__name__)

# Ensure the results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        # Get parameters from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        activation = data.get('activation')
        lr = float(data.get('lr', 0.1))
        step_num = int(data.get('step_num', 1000))

        # Validate parameters
        if activation not in ['tanh', 'relu', 'sigmoid']:
            return jsonify({'error': 'Invalid activation function'}), 400
        
        if not (0.001 <= lr <= 1.0):
            return jsonify({'error': 'Learning rate must be between 0.001 and 1.0'}), 400
        
        if not (100 <= step_num <= 10000):
            return jsonify({'error': 'Steps must be between 100 and 10000'}), 400

        # Run the visualization
        visualize(activation, lr, step_num)

        # Check if visualization was generated
        result_gif = "results/visualize.gif"
        if not os.path.exists(result_gif):
            return jsonify({'error': 'Failed to generate visualization'}), 500

        return jsonify({
            "result_gif": result_gif
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def results(filename):
    try:
        return send_from_directory('results', filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

if __name__ == '__main__':
    app.run(debug=True)