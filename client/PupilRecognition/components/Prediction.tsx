import React, { useState } from "react";
import { View, Text, Button, TextInput, StyleSheet, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";
import { API_URL } from "@env";

type ImagePickerAsset = {
  uri: string;
  type?: string;
  fileName?: string | null;
};

const Prediction = () => {
  const [petName, setPetName] = useState("");
  const [calibrationImages, setCalibrationImages] = useState<
    ImagePickerAsset[]
  >([]);
  const [predictionImage, setPredictionImage] =
    useState<ImagePickerAsset | null>(null);
  const [result, setResult] = useState("");
  const [isCalibrated, setIsCalibrated] = useState(false);

  const handleImageChoice = async (
    setImageFunc: React.Dispatch<
      React.SetStateAction<ImagePickerAsset | ImagePickerAsset[]>
    >,
    isCalibration = false
  ) => {
    Alert.alert("Choose Photo", "Choose how you want to add a photo", [
      {
        text: "Take a photo",
        onPress: () => handleUploadImage(setImageFunc, true, isCalibration),
      },
      {
        text: "Choose from gallery",
        onPress: () => handleUploadImage(setImageFunc, false, isCalibration),
      },
      { text: "Cancel", style: "cancel" },
    ]);
  };

  const handleUploadImage = async (
    setImageFunc: React.Dispatch<
      React.SetStateAction<ImagePickerAsset | ImagePickerAsset[]>
    >,
    fromCamera = false,
    isCalibration = false
  ) => {
    let permissionResult;
    if (fromCamera) {
      permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    } else {
      permissionResult =
        await ImagePicker.requestMediaLibraryPermissionsAsync();
    }

    if (permissionResult.status !== "granted") {
      Alert.alert(
        "Permission Denied",
        "You need to grant permission to use this feature."
      );
      return;
    }

    let result;
    if (fromCamera) {
      result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 1,
      });
    } else {
      result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 1,
      });
    }

    if (!result.canceled) {
      const selectedImage: ImagePickerAsset = {
        uri: result.assets[0].uri,
        type: result.assets[0].type || "image/jpeg",
        fileName:
          result.assets[0].fileName || result.assets[0].uri.split("/").pop(),
      };

      if (isCalibration) {
        if (calibrationImages.length < 5) {
          setCalibrationImages([...calibrationImages, selectedImage]);
        } else {
          Alert.alert("Photo limit", "You can add up to 5 calibration photos.");
        }
      } else {
        setImageFunc(selectedImage);
      }
    }
  };

  const handleCalibration = async () => {
    if (calibrationImages.length < 5) {
      Alert.alert(
        "Low number of photos",
        "You have to add 5 calibration photos."
      );
      return;
    }

    console.log("Calibration in progerss...");
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
      const response = await axios.post(`${API_URL}/calibrate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.status === 200) {
        setIsCalibrated(true);
        Alert.alert("Success", "Model calibrated successfully.");
      }
    } catch (error) {
      console.error(error);
    }
  };

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
    } catch (error) {
      console.error(error);
    }
  };

  const handleClear = () => {
    setResult("");
  };

  return (
    <View style={styles.container}>
      <Text>Pet Name:</Text>
      <TextInput
        style={styles.input}
        value={petName}
        onChangeText={setPetName}
      />

      {!isCalibrated && (
        <>
          <Button
            title="Upload Calibration Photos"
            onPress={() =>
              handleImageChoice(
                (images) => setCalibrationImages(images as ImagePickerAsset[]),
                true
              )
            }
          />
          <Text>
            Liczba dodanych zdjęć kalibracyjnych: {calibrationImages.length}/5
          </Text>
          <Button title="Calibrate" onPress={handleCalibration} />
        </>
      )}

      {isCalibrated && (
        <>
          <Button
            title="Upload photo"
            onPress={() =>
              handleImageChoice((image) =>
                setPredictionImage(image as ImagePickerAsset)
              )
            }
          />
          <Button title="Check!" onPress={handlePrediction} />
        </>
      )}

      {result ? (
        <>
          <Text>{result}</Text>
          <Button title="Clear" onPress={() => handleClear()} />
        </>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: "center",
  },
  input: {
    borderBottomWidth: 1,
    marginBottom: 20,
  },
});

export default Prediction;
