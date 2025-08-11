const express = require("express");
const fs = require("fs");
const path = require("path");
const canvas = require("canvas");
const faceapi = require("face-api.js");
const tf = require('@tensorflow/tfjs'); // Use tfjs for compatibility

const app = express();
const PORT = 3002;

// function base64ToBuffer(base64) {
//   return Buffer.from(base64, 'base64');
// }

function base64ToBuffer(base64String) {
  // Remove data URL prefix if present (e.g., "data:image/png;base64,")
  const matches = base64String.match(/^data:([A-Za-z-+/]+);base64,(.+)$/);
  let base64Data = base64String;
  if (matches) {
    base64Data = matches[2];
  }
  return Buffer.from(base64Data, "base64");
}

// Enable JSON parsing with higher limit for base64 data
app.use(express.json({ limit: "10mb" }));

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Load face-api.js models
const MODELS_URL = path.join(__dirname, "/models");

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(MODELS_URL, 'ssd_mobilenetv1_model')),
  faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(MODELS_URL, 'face_landmark_68_model')),
  faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(MODELS_URL, 'face_recognition_model')),
]).then(() => {
  console.log("Models loaded");
});

// Helper: Convert base64 string to image buffer
function base64ToBuffer(base64) {
  return Buffer.from(base64, 'base64');
}

// ===== /crop-face =====
app.post("/crop-face", async (req, res) => {
  const { image } = req.body;

  if (!image) {
    return res.status(400).json({ message: "No image provided." });
  }

  try {
    const buffer = base64ToBuffer(image);
    const img = await canvas.loadImage(buffer);

    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection || !detection.descriptor) {
      return res.status(400).json({ message: "No face detected in the ID card." });
    }

    // ✅ Save descriptor
    fs.writeFileSync(
      path.join(__dirname, "faces", "descriptor.json"),
      JSON.stringify(Array.from(detection.descriptor))
    );

    // ✅ Save cropped face (optional)
    const regionsToExtract = [detection.detection.box];
    const faceImages = await faceapi.extractFaces(img, regionsToExtract);
    if (faceImages.length > 0) {
      const out = fs.createWriteStream(path.join(__dirname, "faces", "reference.png"));
      const stream = faceImages[0].createPNGStream();
      stream.pipe(out);
    }

    console.log("Reference descriptor saved with length:", detection.descriptor.length);
    res.json({ message: "Face cropped and descriptor saved successfully." });
  } catch (err) {
    console.error("Crop-face error:", err);
    res.status(500).json({ message: "Error processing the ID card." });
  }
});

// ===== /compare-face =====
// app.post("/compare-face", async (req, res) => {
//   const { image } = req.body;
//   const descriptorPath = path.join(__dirname, "faces", "descriptor.json");

//   if (!image) {
//     return res.status(400).json({ message: "No image provided." });
//   }

//   if (!fs.existsSync(descriptorPath)) {
//     return res.status(400).json({ message: "Reference face not found. Please run /crop-face first." });
//   }

//   try {
//     // Load reference descriptor
//     const referenceDescriptorRaw = JSON.parse(
//       fs.readFileSync(descriptorPath)
//     );
//     const referenceDescriptor = new Float32Array(referenceDescriptorRaw);

//     // Load input image
//     const buffer = base64ToBuffer(image);
//     const img = await canvas.loadImage(buffer);

//     const detection = await faceapi
//       .detectSingleFace(img)
//       .withFaceLandmarks()
//       .withFaceDescriptor();

//     if (!detection || !detection.descriptor) {
//       return res.status(400).json({ message: "No face detected in the given image." });
//     }

//     const inputDescriptor = detection.descriptor;

//     // ✅ Check length
//     console.log("Reference descriptor length:", referenceDescriptor.length);
//     console.log("Input descriptor length:", inputDescriptor.length);

//     if (referenceDescriptor.length !== inputDescriptor.length) {
//       return res.status(400).json({
//         message: "Descriptor length mismatch. Cannot compare faces.",
//         inputDescriptorLength: inputDescriptor.length,
//         referenceDescriptorLength: referenceDescriptor.length,
//       });
//     }

//     // ✅ Compare
//     const distance = faceapi.euclideanDistance(referenceDescriptor, inputDescriptor);
//     const threshold = 0.6;
//     const isMatch = distance < threshold;

//     res.json({
//       match: isMatch,
//       similarity: (1 - distance).toFixed(4),
//       distance: distance.toFixed(4),
//     });
//   } catch (err) {
//     console.error("Compare-face error:", err);
//     res.status(500).json({ message: "Internal server error." });
//   }
// });

app.post("/compare-face", async (req, res) => {
  const { image } = req.body;
  const descriptorPath = path.join(__dirname, "faces", "descriptor.json");

  if (!image) {
    return res.status(400).json({ message: "No image provided." });
  }

  if (!fs.existsSync(descriptorPath)) {
    return res.status(400).json({ message: "Reference face not found. Please run /crop-face first." });
  }

  try {
    // Load reference descriptor
    const referenceDescriptorRaw = JSON.parse(
      fs.readFileSync(descriptorPath)
    );
    const referenceDescriptor = new Float32Array(referenceDescriptorRaw);

    // Decode base64 and load image with error handling
    let img;
    try {
      const buffer = base64ToBuffer(image);
      img = await canvas.loadImage(buffer);
    } catch (err) {
      console.error("Error decoding or loading image:", err);
      return res.status(400).json({ message: "Invalid base64 image data." });
    }

    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection || !detection.descriptor) {
      return res.status(400).json({ message: "No face detected in the given image." });
    }

    const inputDescriptor = detection.descriptor;

    console.log("Reference descriptor length:", referenceDescriptor.length);
    console.log("Input descriptor length:", inputDescriptor.length);

    if (referenceDescriptor.length !== inputDescriptor.length) {
      return res.status(400).json({
        message: "Descriptor length mismatch. Cannot compare faces.",
        inputDescriptorLength: inputDescriptor.length,
        referenceDescriptorLength: referenceDescriptor.length,
      });
    }

    const distance = faceapi.euclideanDistance(referenceDescriptor, inputDescriptor);
    const threshold = 0.6;
    const isMatch = distance < threshold;

    res.json({
      match: isMatch,
      similarity: (1 - distance).toFixed(4),
      distance: distance.toFixed(4),
    });
  } catch (err) {
    console.error("Compare-face error:", err);
    res.status(500).json({ message: "Internal server error." });
  }
});



// ===== Start Server =====
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
