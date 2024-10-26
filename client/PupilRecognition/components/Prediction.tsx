import React, { useState } from "react";
import { StyleSheet, Alert } from "react-native";
import { Layout, Text, Button } from "@ui-kitten/components";
import { useImagePicker } from "../hooks/useImagePicker";
import { ImagePickerAsset } from "../types/ImagePickerType";
import PredictionProps from "props/PredictionProps";
import axios from "axios";
import { API_URL } from "@env";

const Prediction: React.FC<PredictionProps> = ({
  petName,
  setProgress,
  setShowProgressBar,
}) => {
  const { handleImageChoice } = useImagePicker();
  const [predictionImage, setPredictionImage] =
    useState<ImagePickerAsset | null>(null);
  const [result, setResult] = useState("");
  const [isCalibrated, setIsCalibrated] = useState(false);

  const handlePrediction = async () => {
    if (!isCalibrated) {
      Alert.alert(
        "Calibration error",
        "You have to calibrate the model before making a prediction."
      );
      return;
    }

    const formData = new FormData();
    formData.append("pet_name", petName);

    if (predictionImage) {
      if (predictionImage.uri) {
        const filename =
          predictionImage.fileName || predictionImage.uri.split("/").pop();
        const match = /\.(\w+)$/.exec(filename || "");
        const type = match ? `image/${match[1]}` : `image/jpeg`;

        formData.append("prediction_image", {
          uri: predictionImage.uri,
          name: filename,
          type: type,
        } as any);
      } else {
        Alert.alert("Error", "Prediction image properties are missing.");
        return;
      }
    } else {
      Alert.alert("No Photo", "You have to add a photo to make a prediction.");
      return;
    }

    try {
      setShowProgressBar(true);
      setProgress(0.5); // Simulate progress
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const distance = response.data.distance;
      console.log(distance);

      const threshold = 0.5;
      const message =
        distance > threshold
          ? `This is your pet ${petName}, Hi ${petName}`
          : "This is not your pet";

      setResult(message);
      setProgress(1); // Complete progress
    } catch (error) {
      console.error(error);
    } finally {
      setShowProgressBar(false);
    }
  };

  const handleClear = () => {
    setResult("");
  };

  return (
    <Layout style={styles.container}>
      {isCalibrated && (
        <>
          <Button
            style={styles.button}
            onPress={() =>
              handleImageChoice((image) =>
                setPredictionImage(image as ImagePickerAsset)
              )
            }
          >
            Upload Photo
          </Button>
          <Button style={styles.button} onPress={handlePrediction}>
            Check!
          </Button>
        </>
      )}

      {result ? (
        <>
          <Text>{result}</Text>
          <Button style={styles.button} onPress={handleClear}>
            Clear
          </Button>
        </>
      ) : null}
    </Layout>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    justifyContent: "center",
  },
  button: {
    marginVertical: 10,
  },
});

export default Prediction;
