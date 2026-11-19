"""Emotion detection using the Watson NLP library (EmotionPredict service)."""
import json

import requests

URL = ('https://sn-watson-emotion.labs.skills.network/v1/'
       'watson.runtime.nlp.v1/NlpService/EmotionPredict')
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness']


def emotion_detector(text_to_analyze):
    """Send text to the Watson emotion service and return a formatted result.

    Returns a dictionary with the scores for anger, disgust, fear, joy and
    sadness plus the 'dominant_emotion'. For blank input (HTTP 400) every
    value in the dictionary is None.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=30)

    if response.status_code == 400:
        result = {emotion: None for emotion in EMOTIONS}
        result['dominant_emotion'] = None
        return result

    formatted = json.loads(response.text)
    scores = formatted['emotionPredictions'][0]['emotion']
    result = {emotion: scores[emotion] for emotion in EMOTIONS}
    result['dominant_emotion'] = max(result, key=result.get)
    return result
