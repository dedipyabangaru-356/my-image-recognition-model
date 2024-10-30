let model;
const classNames = ["Dog", "Cat", "Bird"]; // Replace with your class names

async function loadModel() {
    try {
        // Fetch the model.zip from GitHub
        const response = await fetch('https://dedipyabangaru-356.github.io/my-image-recognition-model/model.zip');
        const arrayBuffer = await response.arrayBuffer();

        // Unzip the file using JSZip
        const zip = await JSZip.loadAsync(arrayBuffer);
        const modelJson = await zip.file("model.json").async("string");
        const weightFiles = await Promise.all(
            Object.keys(zip.files)
                .filter(file => file.endsWith(".bin"))
                .map(async file => {
                    const blob = await zip.file(file).async("blob");
                    return new File([blob], file);
                })
        );

        // Load the model using TensorFlow.js with unzipped model.json and weights
        model = await tf.loadLayersModel(tf.io.browserFiles([new File([modelJson], "model.json"), ...weightFiles]));
        document.getElementById('prediction').innerText = "Model loaded successfully!";
        setupWebcam();
    } catch (error) {
        document.getElementById('prediction').innerText = "Failed to load model.";
        console.error("Model loading error:", error);
    }
}

// Your other functions remain the same
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
