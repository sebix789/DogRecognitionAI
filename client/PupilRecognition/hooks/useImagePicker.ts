import { useState } from "react";
import { Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { ImagePickerAsset } from "../types/ImagePickerType";

export const useImagePicker = () => {
  const [calibrationImages, setCalibrationImages] = useState<
    ImagePickerAsset[]
  >([]);

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

  return {
    calibrationImages,
    setCalibrationImages,
    handleImageChoice,
    handleUploadImage,
  };
};
