"""
SignSpeak Perception Module
Handles real-time landmark extraction using MediaPipe Holistic.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """
    Extracts and normalizes skeletal landmarks from video frames using MediaPipe Holistic.
    Implements pose-centric normalization for scale and position invariance.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initialize MediaPipe Holistic model.
        
        Args:
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: 0 (Lite), 1 (Full), 2 (Heavy)
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        logger.info(f"MediaPipe Holistic initialized (complexity={model_complexity})")
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract landmarks from a single frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Dictionary containing normalized landmarks or None if detection failed
        """
        # Convert BGR to RGB (if needed)
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Process frame
        results = self.holistic.process(frame_rgb)
        
        if results.pose_landmarks is None:
            logger.warning("No pose landmarks detected")
            return None
        
        # Extract and normalize landmarks
        landmarks_dict = {
            'pose': self._extract_pose_landmarks(results.pose_landmarks),
            'left_hand': self._extract_hand_landmarks(results.left_hand_landmarks),
            'right_hand': self._extract_hand_landmarks(results.right_hand_landmarks),
            'face': self._extract_face_landmarks(results.face_landmarks),
            'confidence': self._calculate_confidence(results)
        }
        
        # Apply pose-centric normalization
        normalized_landmarks = self._normalize_landmarks(landmarks_dict)
        
        return normalized_landmarks
    
    def _extract_pose_landmarks(self, pose_landmarks) -> np.ndarray:
        """Extract 33 pose landmarks as (x, y, z, visibility)."""
        if pose_landmarks is None:
            return np.zeros((33, 4))
        
        landmarks = []
        for lm in pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        
        return np.array(landmarks)
    
    def _extract_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract 21 hand landmarks as (x, y, z)."""
        if hand_landmarks is None:
            return np.zeros((21, 3))
        
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        
        return np.array(landmarks)
    
    def _extract_face_landmarks(self, face_landmarks, essential_only: bool = True) -> np.ndarray:
        """
        Extract face landmarks.
        If essential_only=True, extract only ~50 key points (eyes, eyebrows, mouth)
        Otherwise extracts all 468 landmarks.
        """
        if face_landmarks is None:
            num_landmarks = 50 if essential_only else 468
            return np.zeros((num_landmarks, 3))
        
        if essential_only:
            # Essential facial landmark indices
            # Indices for: eyes (33, 133, 362, 263), eyebrows (70, 63, 105, 66, 300, 293, 334, 296),
            # mouth (61, 291, 0, 17, 269, 39, 37, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
            essential_indices = [
                # Left eye
                33, 133, 160, 159, 158, 157, 173, 144, 145, 153,
                # Right eye  
                362, 263, 387, 386, 385, 384, 398, 373, 374, 380,
                # Left eyebrow
                70, 63, 105, 66, 107,
                # Right eyebrow
                300, 293, 334, 296, 336,
                # Mouth
                61, 291, 0, 17, 269, 39, 37, 40, 185, 146, 91, 181, 84, 314, 405, 321, 375
            ]
            
            landmarks = []
            for idx in essential_indices:
                lm = face_landmarks.landmark[idx]
                landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
        
        return np.array(landmarks)
    
    def _calculate_confidence(self, results) -> float:
        """Calculate overall detection confidence."""
        confidences = []
        
        if results.pose_landmarks:
            pose_conf = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
            confidences.append(pose_conf)
        
        # Hands don't have visibility scores, assume 1.0 if detected
        if results.left_hand_landmarks:
            confidences.append(1.0)
        if results.right_hand_landmarks:
            confidences.append(1.0)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _normalize_landmarks(self, landmarks_dict: Dict) -> Dict:
        """
        Apply pose-centric normalization.
        Transform coordinates relative to shoulder midpoint for scale/position invariance.
        """
        pose = landmarks_dict['pose']
        
        # Anchor: midpoint between shoulders (landmarks 11 & 12)
        left_shoulder = pose[11, :3]
        right_shoulder = pose[12, :3]
        anchor = (left_shoulder + right_shoulder) / 2
        
        # Scaling factor: shoulder width
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_width < 0.01:  # Avoid division by zero
            logger.warning("Shoulder width too small, skipping normalization")
            return landmarks_dict
        
        # Normalize pose
        pose_norm = pose.copy()
        pose_norm[:, :3] = (pose[:, :3] - anchor) / shoulder_width
        
        # Normalize hands
        left_hand_norm = (landmarks_dict['left_hand'] - anchor) / shoulder_width if landmarks_dict['left_hand'].any() else landmarks_dict['left_hand']
        right_hand_norm = (landmarks_dict['right_hand'] - anchor) / shoulder_width if landmarks_dict['right_hand'].any() else landmarks_dict['right_hand']
        
        # Normalize face
        face_norm = (landmarks_dict['face'] - anchor) / shoulder_width if landmarks_dict['face'].any() else landmarks_dict['face']
        
        return {
            'pose': pose_norm,
            'left_hand': left_hand_norm,
            'right_hand': right_hand_norm,
            'face': face_norm,
            'confidence': landmarks_dict['confidence'],
            'anchor': anchor,
            'scale': shoulder_width
        }
    
    def serialize_landmarks(self, landmarks_dict: Dict) -> Dict:
        """Convert numpy arrays to JSON-serializable format."""
        return {
            'pose': landmarks_dict['pose'].tolist(),
            'left_hand': landmarks_dict['left_hand'].tolist(),
            'right_hand': landmarks_dict['right_hand'].tolist(),
            'face': landmarks_dict['face'].tolist(),
            'confidence': float(landmarks_dict['confidence']),
            'anchor': landmarks_dict.get('anchor', [0, 0, 0]).tolist() if isinstance(landmarks_dict.get('anchor'), np.ndarray) else [0, 0, 0],
            'scale': float(landmarks_dict.get('scale', 1.0))
        }
    
    def draw_landmarks(self, frame: np.ndarray, landmarks_dict: Dict) -> np.ndarray:
        """
        Draw landmarks on frame for visualization.
        
        Args:
            frame: RGB image
            landmarks_dict: Dictionary of landmarks (unnormalized for proper drawing)
            
        Returns:
            Frame with drawn landmarks
        """
        # This is a placeholder - actual implementation would require MediaPipe results object
        # For now, we'll just return the frame
        return frame
    
    def close(self):
        """Clean up resources."""
        self.holistic.close()
        logger.info("MediaPipe Holistic closed")


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = LandmarkExtractor(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Extract landmarks
            landmarks = extractor.extract_landmarks(frame)
            
            if landmarks:
                print(f"Confidence: {landmarks['confidence']:.2f}")
                print(f"Pose landmarks shape: {landmarks['pose'].shape}")
                print(f"Left hand shape: {landmarks['left_hand'].shape}")
                print(f"Right hand shape: {landmarks['right_hand'].shape}")
                print(f"Face shape: {landmarks['face'].shape}")
                print("---")
            
            # Display frame
            cv2.imshow('SignSpeak - Landmark Extraction', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
