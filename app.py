import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from twilio.rest import Client

app = Flask(__name__)

# ---------------------- Load Image Embedding Model ----------------------
model = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    trainable=False,
    input_shape=(224, 224, 3)
)

# ---------------------- Twilio Configuration ----------------------
TWILIO_PHONE_NUMBER = +13204416458  # Your Twilio phone number
TO_PHONE_NUMBER = +917981363612    # The recipient's phone number
ACCOUNT_SID = 'ACac6d1553bdcab6d877f28f2cd6e6aecf'  # Twilio Account SID
AUTH_TOKEN = '89dba5c81b19c2b7988bc77a1516f853'     # Twilio Auth Token
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ---------------------- Helper Functions ----------------------
def preprocess_image(image_data):
    """Preprocess image data for the embedding model."""
    img = tf.image.decode_jpeg(image_data, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def extract_embeddings(image_data_list, model):
    """Extract embeddings for a list of images."""
    embeddings = []
    for image_data in image_data_list:
        preprocessed_img = preprocess_image(image_data)
        embedding = model(tf.expand_dims(preprocessed_img, 0))
        embeddings.append(np.array(embedding))
    return embeddings

def find_most_similar_image(uploaded_image_data, dataset_embeddings, dataset_image_paths):
    """Find the most similar image based on embeddings."""
    uploaded_image_embedding = extract_embeddings([uploaded_image_data], model)[0]

    similarities = []
    for dataset_embedding in dataset_embeddings:
        similarity = np.dot(uploaded_image_embedding.flatten(), dataset_embedding.flatten())
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)
    return dataset_image_paths[most_similar_index], float(similarities[most_similar_index])

def image_to_base64(image_path):
    """Convert image to Base64 for embedding in HTML."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def send_sms(message):
    """Send SMS notification using Twilio."""
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=TO_PHONE_NUMBER
        )
        print(f"Message sent: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {str(e)}")


# ---------------------- Routes ----------------------

@app.route("/")
def signup():
    """Render the Sign-Up Page."""
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Render the Login Page."""
    if request.method == "POST":
        # Here, you'd typically validate user credentials.
        # For now, redirect to Home on any POST request.
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/home")
def home():
    """Render the Home Page."""
    return render_template("home.html")

@app.route('/second')
def second():
    return render_template('second.html')


@app.route("/index", methods=["GET", "POST"])
def upload_image():
    """Handle image upload and similarity search."""
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            uploaded_image_data = uploaded_file.read()
            
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            # Build the path to the images folder relative to BASE_DIR
            dataset_image_dir = os.path.join(BASE_DIR, 'static', 'images')
            dataset_image_paths = [
                os.path.join(dataset_image_dir, img)
                for img in os.listdir(dataset_image_dir)
                if os.path.isfile(os.path.join(dataset_image_dir, img))
            ]
            
            # Preprocess and compute embeddings
            dataset_images_data = [open(path, 'rb').read() for path in dataset_image_paths]
            dataset_embeddings = extract_embeddings(dataset_images_data, model)

            # Find most similar image
            most_similar_image_path, similarity_score = find_most_similar_image(
                uploaded_image_data, dataset_embeddings, dataset_image_paths
            )
            
            # Check similarity threshold
            similarity_threshold = 500  # Adjustable threshold
            if similarity_score < similarity_threshold:
                message = f"Oops! No similar image found. Similarity score: {similarity_score:.2f}"
                #send_sms(message)  # Send notification
                return jsonify({
                    "below_threshold": True,
                    "similarity_score": similarity_score
                })


            # Convert the matched image to Base64
            similar_image_base64 = image_to_base64(most_similar_image_path)
            return jsonify({
                "most_similar_image": f"data:image/jpeg;base64,{similar_image_base64}",
                "similarity_score": similarity_score
            })

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
