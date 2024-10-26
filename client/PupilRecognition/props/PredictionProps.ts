export default interface PredictionProps {
  petName: string;
  setProgress: React.Dispatch<React.SetStateAction<number>>;
  setShowProgressBar: React.Dispatch<React.SetStateAction<boolean>>;
}
