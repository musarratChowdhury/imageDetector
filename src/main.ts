import * as mobilenet from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // Register WebGL backend

let predictionsDiv = document.getElementById("predictions") as HTMLDivElement;

// Image classification function
async function classifyImage(imgElement: HTMLImageElement) {
  // Set the backend to WebGL
  await tf.setBackend("webgl");
  await tf.ready();
  console.log("Active backend:", tf.getBackend());

  // Load the MobileNet model
  const model = await mobilenet.load({
    version: 1,
    alpha: 0.25,
  });

  // Classify the image
  const predictions = await model.classify(imgElement);

  // Show predictions in the UI

  predictionsDiv.innerHTML = "";
  predictions.forEach((prediction) => {
    const p = document.createElement("p");
    p.innerText = `Class: ${prediction.className}, Probability: ${(
      prediction.probability * 100
    ).toFixed(2)}%`;
    predictionsDiv.appendChild(p);
  });
}

// Handle image upload and preview
const fileInput = document.getElementById("fileInput") as HTMLInputElement;
const imgPreview = document.getElementById("imgPreview") as HTMLImageElement;
let imgElement: any = null;

fileInput.addEventListener("change", (event: any) => {
  const file = event.target!.files[0];
  predictionsDiv.innerHTML = "";
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imgPreview.src = `${e.target!.result}`;
      imgPreview.style.display = "block";
      imgElement = new Image();
      imgElement.src = e.target!.result;
      imgElement.width = 224; // MobileNet model expects 224x224 images
      imgElement.height = 224;
    };
    reader.readAsDataURL(file);
  }
});

// Handle detect button click
const detectButton = document.getElementById(
  "detectButton"
) as HTMLButtonElement;
detectButton!.addEventListener("click", () => {
  if (imgElement) {
    classifyImage(imgElement); // Call the classify function
  } else {
    alert("Please upload an image first.");
  }
});
