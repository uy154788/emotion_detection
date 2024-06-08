from flask import Flask, request, jsonify
import cv2
import dlib
from fer import FER
import numpy as np
import requests

app = Flask(__name__)

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")  # Ensure this file is in your working directory

# Initialize the FER emotion detector
emotion_detector = FER()


@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    print("Received data:", data)
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    # Download the video file from the provided URL
    video_path = 'temp_video.mp4'
    response = requests.get(video_url)
    with open(video_path, 'wb') as f:
        f.write(response.content)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Emotion scores accumulator
    emotion_scores = {
        'angry': [],
        'disgust': [],
        'fear': [],
        'happy': [],
        'sad': [],
        'surprise': [],
        'neutral': []
    }

    frame_count = 0
    frame_skip = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            # Convert frame to grayscale for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = detector(gray)

            # Loop through detected faces
            for face in faces:
                # Detect facial landmarks
                landmarks = predictor(gray, face)

                # Extract face region
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_region = frame[y:y + h, x:x + w]

                # Detect emotion using the FER library
                emotions = emotion_detector.detect_emotions(face_region)
                if emotions:
                    # Accumulate scores for each emotion
                    for emotion, score in emotions[0]['emotions'].items():
                        emotion_scores[emotion].append(score)

    # Release resources
    cap.release()

    # Calculate average emotion scores
    average_emotion_scores = {emotion: np.mean(scores) if scores else 0 for emotion, scores in emotion_scores.items()}
    for emotion, score in average_emotion_scores.items():
        average_emotion_scores[emotion] = round(average_emotion_scores[emotion] * frame_count)

    # Calculate final confidence
    confidence_level = 0
    for emotion, score in average_emotion_scores.items():
        if emotion in ['happy']:
            confidence_level += 1 * average_emotion_scores[emotion]
        elif emotion in ['neutral']:
            confidence_level += 0.8 * average_emotion_scores[emotion]
        elif emotion in ['surprise']:
            confidence_level += 0.6 * average_emotion_scores[emotion]
        elif emotion in ['sad', 'fear']:
            confidence_level += 0.4 * average_emotion_scores[emotion]
        elif emotion in ['angry', 'disgust']:
            confidence_level += 0.1 * average_emotion_scores[emotion]

    confidence_level = round((confidence_level / frame_count) * 100, 2)

    # Return results
    return jsonify({
        "average_emotion_scores": average_emotion_scores,
        "frame_count": frame_count,
        "confidence_level": confidence_level
    })


if __name__ == '__main__':
    app.run(debug=False)
