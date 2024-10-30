let model;
let net = new brain.NeuralNetwork();


async function loadModel() {
    try {
        model = await tf.loadLayersModel('model/model.json');
        document.getElementById('prediction').innerText = "Model loaded successfully!";
        setupWebcam();
    } catch (error) {
        document.getElementById('prediction').innerText = "Failed to load model.";
        console.error("Model loading error:", error);
    }
}

async function setupWebcam() {
    const webcamElement = document.getElementById('webcam');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamElement.srcObject = stream;
    predictWithBrainJS();
}



async function predictWithBrainJS() {
    const webcamElement = document.getElementById('webcam');
    const predictionElement = document.getElementById('prediction');

    while (true) {
        const imageCapture = tf.browser.fromPixels(webcamElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .expandDims();

        if (model) {
            const predictions = await model.predict(imageCapture).data();
            
            console.log("Predictions Array: ", predictions); // Debug: Check the predictions array

            let maxProbability = 0;
            let predictedClassIndex = -1;
            predictions.forEach((probability, index) => {
                if (probability > maxProbability) {
                    maxProbability = probability;
                    predictedClassIndex = index;
                }
            });

            console.log("Predicted Class Index: ", predictedClassIndex); // Debug: Check which class is predicted

            const predictedClass = classNames[predictedClassIndex] || "Unknown";
            predictionElement.innerText = `Prediction: ${predictedClass} (Confidence: ${(maxProbability * 100).toFixed(2)}%)`;
        }

        imageCapture.dispose();
        await tf.nextFrame();
    }
}


const webcamElement = document.getElementById('webcam');
const predictionElement = document.getElementById('prediction');

async function loadModel() {
    const model = await tf.loadLayersModel('model/model.json'); // Adjust the path to your model
    console.log('Model loaded successfully!');
    return model;
}

async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamElement.srcObject = stream;
}

async function predict(model) {
    const webcamImage = tf.browser.fromPixels(webcamElement);
    const resizedImage = tf.image.resizeBilinear(webcamImage, [224, 224]);
    const normalizedImage = resizedImage.div(255);
    const input = normalizedImage.expandDims(0); // Make it a batch of 1

    const predictions = model.predict(input);
    const predictionData = await predictions.data();

    const highestPrediction = Math.max(...predictionData);
    const predictedClass = predictionData.indexOf(highestPrediction);
    const confidence = (highestPrediction * 100).toFixed(2);

    predictionElement.innerText = `Prediction: Class ${predictedClass} (Matching: ${confidence}%)`;

    webcamImage.dispose();
    resizedImage.dispose();
    normalizedImage.dispose();
    input.dispose();
}

async function main() {
    await startWebcam();
    const model = await loadModel();
    
    setInterval(() => predict(model), 1000);
}

main();

loadModel();
