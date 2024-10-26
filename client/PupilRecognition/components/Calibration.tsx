import React, { useState } from "react";
import { StyleSheet, Alert } from "react-native";
import { Layout, Text, Button } from "@ui-kitten/components";
import { useImagePicker } from "../hooks/useImagePicker";
import { ImagePickerAsset } from "../types/ImagePickerType";
import CalibrationProps from "props/CalibrationProps";
import axios from "axios";
import { API_URL } from "@env";

const Calibration: React.FC<CalibrationProps> = ({
  petName,
  setProgress,
  setShowProgressBar,
}) => {
  const { calibrationImages, handleImageChoice, setCalibrationImages } =
    useImagePicker();
  const [isCalibrated, setIsCalibrated] = useState(false);

  const handleCalibration = async () => {
    if (calibrationImages.length < 5) {
      Alert.alert(
        "Low number of photos",
        "You have to add 5 calibration photos."
      );
      return;
    }

    console.log("Calibration in progress...");
    const formData = new FormData();
    formData.append("pet_name", petName);

    calibrationImages.forEach((image) => {
      if (image.uri) {
        const filename = image.fileName || image.uri.split("/").pop();
        const match = /\.(\w+)$/.exec(filename || "");
        const type = match ? `image/${match[1]}` : `image/jpeg`;

        formData.append("calibration_images", {
          uri: image.uri,
          name: filename,
          type: type,
        } as any);
      } else {
        console.error("Image properties are missing");
      }
    });

    try {
      setShowProgressBar(true);
      setProgress(0.5);
      const response = await axios.post(`${API_URL}/calibrate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.status === 200) {
        setIsCalibrated(true);
        setProgress(1);
        Alert.alert("Success", "Model calibrated successfully.");
      }
    } catch (error) {
      console.error(error);
    } finally {
      setShowProgressBar(false);
    }
  };

  return (
    <Layout style={styles.container}>
      {!isCalibrated && (
        <>
          <Button
            style={styles.button}
            onPress={() =>
              handleImageChoice(
                (images) => setCalibrationImages(images as ImagePickerAsset[]),
                true
              )
            }
          >
            Upload Calibration Photos
          </Button>
          <Text style={styles.text}>
            Number of calibration photos added: {calibrationImages.length}/5
          </Text>
          <Button style={styles.button} onPress={handleCalibration}>
            Calibrate
          </Button>
        </>
      )}
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
  text: {
    margin: 20,
    marginTop: 5,
  },
});

export default Calibration;
