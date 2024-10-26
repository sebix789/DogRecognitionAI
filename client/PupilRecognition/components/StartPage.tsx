import React, { useState } from "react";
import { StyleSheet } from "react-native";
import { Layout, Text, Input, Button } from "@ui-kitten/components";
import StartPageProps from "../props/StartPageProps";

const StartPage: React.FC<StartPageProps> = ({ onSubmit }) => {
  const [petName, setPetName] = useState("");

  return (
    <Layout style={styles.container}>
      <Text category="h5">Pet Name:</Text>
      <Input style={styles.input} value={petName} onChangeText={setPetName} />
      <Button style={styles.button} onPress={() => onSubmit(petName)}>
        Submit
      </Button>
    </Layout>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: "center",
  },
  input: {
    marginBottom: 20,
  },
  button: {
    marginVertical: 10,
  },
});

export default StartPage;
