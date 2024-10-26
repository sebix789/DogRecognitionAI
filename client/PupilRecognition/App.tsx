import React, { useState } from "react";
import {
  ApplicationProvider,
  Icon,
  IconRegistry,
  Layout,
  Toggle,
  ProgressBar,
} from "@ui-kitten/components";
import { EvaIconsPack } from "@ui-kitten/eva-icons";
import * as eva from "@eva-design/eva";
import StartPage from "./components/StartPage";
import Calibration from "./components/Calibration";
import Prediction from "./components/Prediction";

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [currentPage, setCurrentPage] = useState("StartPage");
  const [petName, setPetName] = useState("");
  const [progress, setProgress] = useState(0);
  const [showProgressBar, setShowProgressBar] = useState(false);

  const handlePageChange = (page: string, name = "") => {
    setPetName(name);
    setCurrentPage(page);
  };

  return (
    <>
      <IconRegistry icons={EvaIconsPack} />
      <ApplicationProvider {...eva} theme={darkMode ? eva.dark : eva.light}>
        <Layout style={{ flex: 1 }}>
          <Toggle
            style={{ alignSelf: "flex-end", margin: 80 }}
            checked={darkMode}
            onChange={() => setDarkMode(!darkMode)}
          >
            {darkMode ? "Dark Mode" : "Light Mode"}
          </Toggle>
          {currentPage === "StartPage" && (
            <StartPage onSubmit={(name) => handlePageChange("Main", name)} />
          )}
          {currentPage === "Main" && (
            <>
              <Layout style={{ display: "flex", flexDirection: "column" }}>
                <Calibration
                  petName={petName}
                  setProgress={setProgress}
                  setShowProgressBar={setShowProgressBar}
                />
                <Prediction
                  petName={petName}
                  setProgress={setProgress}
                  setShowProgressBar={setShowProgressBar}
                />
              </Layout>
              {showProgressBar && (
                <ProgressBar progress={progress} style={{ marginTop: 10 }} />
              )}
            </>
          )}
        </Layout>
      </ApplicationProvider>
    </>
  );
}
