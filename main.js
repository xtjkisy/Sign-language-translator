import { KNNImageClassifier } from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';

const IMAGE_SIZE = 227;
const TOPK = 10;
const confidenceThreshold = 0.98;
const words = ["start", "stop"];

class Main {
  constructor() {
    this.knn = null;
    this.video = document.getElementById("video");
    this.predicting = false;
    this.previousPrediction = -1;

    this.initializeUI();
    this.setupWebcam();
    this.loadKNNModel();
  }

  initializeUI() {
    this.proceedBtn = document.getElementById("proceedButton");
    this.predButton = document.getElementById("predictButton");
    this.backToTrainButton = document.getElementById("backButton");
    this.translationText = document.getElementById("translationText");

    this.proceedBtn.addEventListener('click', () => {
      this.proceedToTraining();
    });

    this.predButton.addEventListener('click', () => {
      if (this.knn) {
        this.togglePrediction();
      } else {
        console.error('KNN model is not loaded.');
      }
    });

    this.backToTrainButton.addEventListener('click', () => {
      this.stopPrediction();
      this.showTrainingUI();
    });

    this.predButton.style.display = "none";
    this.backToTrainButton.style.display = "none";
  }

  setupWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;
        this.video.addEventListener('playing', () => {
          console.log("Video is playing");
        });
      })
      .catch((error) => {
        console.error('Error accessing webcam:', error);
      });
  }

  loadKNNModel() {
    this.knn = new KNNImageClassifier(words.length, TOPK);

    this.knn.load()
      .then(() => {
        console.log("KNN model loaded successfully");
        this.initialTraining();
      })
      .catch((error) => {
        console.error('Error loading KNN model:', error);
      });
  }

  initialTraining() {
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");

    startButton.addEventListener('click', () => {
      this.trainGesture(0); // Train start gesture
    });

    stopButton.addEventListener('click', () => {
      this.trainGesture(1); // Train stop gesture
    });

    console.log("Initial training setup completed");
  }

  trainGesture(index) {
    if (this.video.srcObject) {
      const image = dl.fromPixels(this.video);
      this.knn.addImage(image, index);
      image.dispose();

      const exampleCount = this.knn.getClassExampleCount()[index];
      console.log(`Added example for ${words[index]}. Total examples: ${exampleCount}`);

      // Update UI display
      // (You can add UI updates to show example count and training progress)
    } else {
      console.error('Webcam stream not initialized.');
    }
  }

  proceedToTraining() {
    const welcomeContainer = document.getElementById("welcomeContainer");
    const trainingContainer = document.getElementById("trainingContainer");

    welcomeContainer.style.display = "none";
    trainingContainer.style.display = "block";
  }

  togglePrediction() {
    this.predicting = !this.predicting;
    if (this.predicting) {
      this.predButton.innerText = "Stop Prediction";
      this.startPrediction();
    } else {
      this.stopPrediction();
    }
  }

  startPrediction() {
    this.predictionLoop();
    this.predButton.style.display = "none";
    this.backToTrainButton.style.display = "block";
  }

  stopPrediction() {
    cancelAnimationFrame(this.predictionLoop);
    this.predButton.innerText = "Translate";
    this.predButton.style.display = "block";
    this.backToTrainButton.style.display = "none";
  }

  predictionLoop() {
    if (this.predicting) {
      const image = dl.fromPixels(this.video);


      this.knn.predictClass(image)
        .then((result) => {
          const { classIndex, confidences } = result;
          if (confidences[classIndex] > confidenceThreshold && classIndex !== this.previousPrediction) {
            const predictedWord = words[classIndex];
            console.log(`Predicted word: ${predictedWord}`);
            this.translationText.innerText = predictedWord;
            this.previousPrediction = classIndex;
          }
        })
        .then(() => image.dispose())
        .then(() => requestAnimationFrame(this.predictionLoop.bind(this)));
    }
  }

  showTrainingUI() {
    const trainingContainer = document.getElementById("trainingContainer");
    const translateContainer = document.getElementById("translateContainer");

    trainingContainer.style.display = "block";
    translateContainer.style.display = "none";
  }
}

// Instantiate Main class when DOM content is loaded
document.addEventListener('DOMContentLoaded', () => {
  const main = new Main();
});
