export default interface CalibrationProps {
  petName: string;
  setProgress: React.Dispatch<React.SetStateAction<number>>;
  setShowProgressBar: React.Dispatch<React.SetStateAction<boolean>>;
}
