import os
import torch
import numpy as np
import pandas as pd
import librosa
import pickle
from torch.nn import functional as F
import matplotlib.pyplot as plt
from train import ImprovedEmotionCNN
from extract_features import extract_features

def load_model_and_metadata(model_dir="models"):
    """Load the trained model and necessary metadata"""
    # Load class mapping
    mapping_path = os.path.join(model_dir, "class_mapping.csv")
    class_mapping = pd.read_csv(mapping_path)
    class_names = class_mapping['emotion'].values
    
    # Load feature scaler
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Determine input size from scaler
    input_size = scaler.mean_.shape[0]
    
    # Load best model
    model_path = os.path.join(model_dir, "best_emotion_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_emotion_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "emotion_model.pth")  # Original model filename
    
    # Initialize model
    model = ImprovedEmotionCNN(input_size, len(class_names))
    
    # Check if using checkpoint format or direct state dict
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', 'unknown')
        best_val_acc = checkpoint.get('val_acc', 'unknown')
        print(f"Loaded model from epoch {best_epoch} with validation accuracy {best_val_acc}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    
    model.eval()
    
    return model, class_names, scaler

def predict_emotion(audio_path, model, class_names, scaler):
    """Predict emotion from audio file"""
    # Extract features
    features = extract_features(audio_path, augment=False)
    
    if features is None:
        return "Error processing audio file"
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Convert to tensor and reshape for CNN
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Get probabilities for all classes
    class_probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return {
        'predicted_emotion': class_names[predicted_class],
        'confidence': float(probabilities[predicted_class]),
        'all_probabilities': class_probs
    }

def test_on_folder(test_folder, model, class_names, scaler):
    """Test model on all audio files in a folder"""
    results = []
    
    if not os.path.exists(test_folder):
        print(f"Test folder not found: {test_folder}")
        return results
    
    for filename in os.listdir(test_folder):
        if filename.endswith(('.wav', '.mp3')):
            file_path = os.path.join(test_folder, filename)
            
            # Extract true label from filename if available
            true_label = None
            for emotion in class_names:
                if emotion in filename.lower():
                    true_label = emotion
                    break
            
            try:
                result = predict_emotion(file_path, model, class_names, scaler)
                result['file'] = filename
                result['true_label'] = true_label
                results.append(result)
                
                print(f"File: {filename}")
                print(f"Predicted emotion: {result['predicted_emotion']}")
                print(f"Confidence: {result['confidence']:.4f}")
                if true_label:
                    correct = result['predicted_emotion'] == true_label
                    print(f"True emotion: {true_label} ({'✓' if correct else '✗'})")
                print("-" * 40)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Calculate accuracy if true labels are available
    labeled_results = [r for r in results if r['true_label'] is not None]
    if labeled_results:
        correct = sum(1 for r in labeled_results if r['predicted_emotion'] == r['true_label'])
        accuracy = correct / len(labeled_results)
        print(f"\nOverall accuracy: {accuracy:.4f} ({correct}/{len(labeled_results)})")
        
        # Class-wise accuracy
        class_correct = {emotion: 0 for emotion in class_names}
        class_total = {emotion: 0 for emotion in class_names}
        
        for r in labeled_results:
            true_label = r['true_label']
            class_total[true_label] += 1
            if r['predicted_emotion'] == true_label:
                class_correct[true_label] += 1
        
        print("\nClass-wise accuracy:")
        for emotion in class_names:
            if class_total[emotion] > 0:
                acc = class_correct[emotion] / class_total[emotion]
                print(f"{emotion}: {acc:.4f} ({class_correct[emotion]}/{class_total[emotion]})")
        
        # Create confusion matrix
        confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
        for r in labeled_results:
            true_idx = list(class_names).index(r['true_label'])
            pred_idx = list(class_names).index(r['predicted_emotion'])
            confusion[true_idx, pred_idx] += 1
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = confusion.max() / 2
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                plt.text(j, i, format(confusion[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.savefig("confusion_matrix.png")
        print("\nConfusion matrix saved to confusion_matrix.png")
    
    return results

def plot_emotion_probabilities(probabilities, title="Emotion Prediction"):
    """Plot emotion probabilities as a bar chart"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    
    bars = plt.bar(emotions, probs, color=colors)
    
    # Highlight the highest probability
    max_idx = probs.index(max(probs))
    bars[max_idx].set_color('red')
    
    plt.title(title)
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig("emotion_prediction.png")
    print("Emotion probability chart saved to emotion_prediction.png")
    plt.close()

if __name__ == "__main__":
    import sys
    
    model, class_names, scaler = load_model_and_metadata()
    
    if len(sys.argv) == 1:
        # No arguments - interactive mode
        while True:
            print("\nOptions:")
            print("1. Test single audio file")
            print("2. Test folder of audio files")
            print("3. Exit")
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                audio_path = input("Enter path to audio file: ")
                if os.path.exists(audio_path):
                    result = predict_emotion(audio_path, model, class_names, scaler)
                    print(f"\nPredicted emotion: {result['predicted_emotion']}")
                    print(f"Confidence: {result['confidence']:.4f}")
                    plot_emotion_probabilities(result['all_probabilities'], 
                                           f"Emotion Prediction for {os.path.basename(audio_path)}")
                else:
                    print(f"File not found: {audio_path}")
            
            elif choice == '2':
                folder_path = input("Enter path to folder with audio files: ")
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    test_on_folder(folder_path, model, class_names, scaler)
                else:
                    print(f"Folder not found: {folder_path}")
            
            elif choice == '3':
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    elif len(sys.argv) == 2:
        # One argument - could be file or folder
        path = sys.argv[1]
        
        if os.path.isdir(path):
            # Test all files in folder
            test_on_folder(path, model, class_names, scaler)
        
        elif os.path.isfile(path):
            # Test single file
            result = predict_emotion(path, model, class_names, scaler)
            print(f"\nPredicted emotion: {result['predicted_emotion']}")
            print(f"Confidence: {result['confidence']:.4f}")
            plot_emotion_probabilities(result['all_probabilities'], 
                                   f"Emotion Prediction for {os.path.basename(path)}")
        
        else:
            print(f"Path not found: {path}")
    
    else:
        print("Usage: python test.py [audio_file_or_folder]")
        print("       If no arguments provided, interactive mode will be used.")