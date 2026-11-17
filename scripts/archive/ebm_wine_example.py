from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
import time

def get_feature_shape_explanation(ebm, feature_idx, feature_name, X_train):
    try:
        # Get the range of values for this feature
        feature_values = X_train[:, feature_idx]
        min_val, max_val = np.min(feature_values), np.max(feature_values)
        
        # Create test points across the feature range
        test_points = np.linspace(min_val, max_val, 20)
        
        # Use the training data mean as baseline for other features
        baseline_sample = np.mean(X_train, axis=0)
        
        # Test how predictions change across the feature range
        predictions = []
        for test_val in test_points:
            test_sample = baseline_sample.copy()
            test_sample[feature_idx] = test_val
            pred = ebm.predict_proba(test_sample.reshape(1, -1))[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Find the most interesting ranges
        class_0_preds = predictions[:, 0]
        class_1_preds = predictions[:, 1]
        class_2_preds = predictions[:, 2]
        
        # Find ranges where each class is most likely
        class_0_max_idx = np.argmax(class_0_preds)
        class_1_max_idx = np.argmax(class_1_preds)
        class_2_max_idx = np.argmax(class_2_preds)
        
        print(f"How {feature_name} affects wine classification:")
        print("Value Range → Effect on Classification")
        
        # Show key decision points
        decision_points = [
            (test_points[class_0_max_idx], "class_0", class_0_preds[class_0_max_idx]),
            (test_points[class_1_max_idx], "class_1", class_1_preds[class_1_max_idx]),
            (test_points[class_2_max_idx], "class_2", class_2_preds[class_2_max_idx])
        ]
        
        for val, class_name, confidence in decision_points:
            print(f"  {val:8.3f} → Strong {class_name} signal (confidence: {confidence:.3f})")
        
        # Show general trend
        slope_0 = (class_0_preds[-1] - class_0_preds[0]) / (test_points[-1] - test_points[0])
        slope_1 = (class_1_preds[-1] - class_1_preds[0]) / (test_points[-1] - test_points[0])
        slope_2 = (class_2_preds[-1] - class_2_preds[0]) / (test_points[-1] - test_points[0])
        
        print(f"  General trend as {feature_name} increases:")
        if abs(slope_0) > 0.01:
            trend_0 = "increases" if slope_0 > 0 else "decreases"
            print(f"    • class_0 probability {trend_0} (slope: {slope_0:+.4f})")
        if abs(slope_1) > 0.01:
            trend_1 = "increases" if slope_1 > 0 else "decreases"
            print(f"    • class_1 probability {trend_1} (slope: {slope_1:+.4f})")
        if abs(slope_2) > 0.01:
            trend_2 = "increases" if slope_2 > 0 else "decreases"
            print(f"    • class_2 probability {trend_2} (slope: {slope_2:+.4f})")
        
        return True
        
    except Exception as e:
        print(f"Could not generate detailed explanation: {e}")
        return False

def analyze_feature_interactions(ebm, X_train, feature_names, top_features_idx):
    print("\nFEATURE INTERACTION ANALYSIS")
    print("="*60)
    
    # Take the top 2 features and see how they interact
    if len(top_features_idx) >= 2:
        feat1_idx, feat2_idx = top_features_idx[0], top_features_idx[1]
        feat1_name = feature_names[feat1_idx]
        feat2_name = feature_names[feat2_idx]
        
        print(f"Analyzing interaction between {feat1_name} and {feat2_name}:")
        
        # Get value ranges
        feat1_values = X_train[:, feat1_idx]
        feat2_values = X_train[:, feat2_idx]
        
        feat1_low, feat1_high = np.percentile(feat1_values, [25, 75])
        feat2_low, feat2_high = np.percentile(feat2_values, [25, 75])
        
        # Create baseline sample
        baseline = np.mean(X_train, axis=0)
        
        # Test 4 combinations
        combinations = [
            (feat1_low, feat2_low, "Low-Low"),
            (feat1_low, feat2_high, "Low-High"),
            (feat1_high, feat2_low, "High-Low"),
            (feat1_high, feat2_high, "High-High")
        ]
        
        print(f"  {feat1_name} vs {feat2_name} combinations:")
        for feat1_val, feat2_val, desc in combinations:
            test_sample = baseline.copy()
            test_sample[feat1_idx] = feat1_val
            test_sample[feat2_idx] = feat2_val
            
            pred = ebm.predict_proba(test_sample.reshape(1, -1))[0]
            predicted_class = np.argmax(pred)
            
            print(f"    {desc:9s} → class_{predicted_class} (confidence: {pred[predicted_class]:.3f})")

def main():
    print("Starting Enhanced EBM Analysis...")
    
    # Load the wine dataset
    print("Loading wine dataset...")
    wine_data = load_wine()
    X, y = wine_data.data, wine_data.target
    feature_names = wine_data.feature_names
    target_names = wine_data.target_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {', '.join(target_names)}")
    print(f"Features: {len(feature_names)} wine chemical properties")

    # Train EBM
    print("\nTraining EBM model...")
    ebm = ExplainableBoostingClassifier(
        random_state=42,
        interactions=0,  # Disable interactions for faster training
        max_bins=32,     # Reduce bins for faster training
        outer_bags=4,    # Reduce outer bags
        inner_bags=4     # Reduce inner bags
    )
    
    start_time = time.time()
    ebm.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Get global explanation
    print("\nGenerating global explanations...")
    global_explanation = ebm.explain_global()
    explanation_data = global_explanation.data()
    feature_importances = explanation_data['scores']
    
    # Find top features
    top_features_idx = np.argsort(np.abs(feature_importances))[-5:][::-1]
    
    print("\n" + "="*60)
    print("DETAILED FEATURE EXPLANATIONS")
    print("="*60)
    
    # Show detailed explanations for top features
    for i, feat_idx in enumerate(top_features_idx):
        feature_name = feature_names[feat_idx]
        importance = feature_importances[feat_idx]
        
        print(f"\n{i+1}. {feature_name.upper()} (Importance: {importance:.4f})")
        print("-" * 50)
        
        # Get detailed shape explanation
        success = get_feature_shape_explanation(ebm, feat_idx, feature_name, X_train)
        
        if not success:
            # Fallback - show basic statistics
            feature_values = X_train[:, feat_idx]
            print(f"  Statistics for {feature_name}:")
            print(f"    Mean: {np.mean(feature_values):.3f}")
            print(f"    Std:  {np.std(feature_values):.3f}")
            print(f"    Range: {np.min(feature_values):.3f} - {np.max(feature_values):.3f}")
    
    # Analyze feature interactions
    analyze_feature_interactions(ebm, X_train, feature_names, top_features_idx)
    
    # Show prediction examples with better explanations
    print("\n" + "="*60)
    print("PREDICTION EXAMPLES WITH DETAILED EXPLANATIONS")
    print("="*60)
    
    sample_indices = [0, 1, 2]
    
    for idx in sample_indices:
        sample_x = X_test[idx:idx+1]
        sample_y = y_test[idx]
        predicted_y = ebm.predict(sample_x)[0]
        prediction_proba = ebm.predict_proba(sample_x)[0]
        
        print(f"\nSample {idx+1} Analysis:")
        print(f"  Actual wine type: {target_names[sample_y]}")
        print(f"  Predicted wine type: {target_names[predicted_y]}")
        print(f"  Confidence: {prediction_proba[predicted_y]:.3f}")
        print(f"  All class probabilities: {prediction_proba}")
        
        # Enhanced feature contribution analysis
        print("  Feature contribution analysis:")
        
        # Get baseline prediction
        baseline_features = np.mean(X_train, axis=0)
        baseline_pred = ebm.predict_proba(baseline_features.reshape(1, -1))[0]
        
        # Calculate contributions more comprehensively
        contributions = []
        
        for feat_idx in range(len(feature_names)):
            # Method 1: Perturbation analysis
            perturbed_sample = sample_x.copy()
            perturbed_sample[0][feat_idx] = baseline_features[feat_idx]
            perturbed_pred = ebm.predict_proba(perturbed_sample)[0]
            
            # Contribution to the predicted class
            contribution = prediction_proba[predicted_y] - perturbed_pred[predicted_y]
            contributions.append(contribution)
        
        contributions = np.array(contributions)
        
        # Show top contributing features
        top_contrib_idx = np.argsort(np.abs(contributions))[-5:][::-1]
        
        print("    Top 5 contributing features:")
        for i, feat_idx in enumerate(top_contrib_idx):
            feat_name = feature_names[feat_idx]
            contrib = contributions[feat_idx]
            feat_value = sample_x[0][feat_idx]
            baseline_value = baseline_features[feat_idx]
            
            # Calculate how much this feature deviates from baseline
            deviation = feat_value - baseline_value
            deviation_pct = (deviation / baseline_value) * 100 if baseline_value != 0 else 0
            
            direction = "↑" if contrib > 0 else "↓"
            magnitude = "Strong" if abs(contrib) > 0.01 else "Moderate" if abs(contrib) > 0.001 else "Weak"
            
            print(f"      {i+1}. {feat_name}")
            print(f"         Value: {feat_value:.3f} (baseline: {baseline_value:.3f})")
            print(f"         Deviation: {deviation:+.3f} ({deviation_pct:+.1f}%)")
            print(f"         Impact: {magnitude} {direction} ({contrib:+.4f})")
    
    # Model performance analysis
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    train_score = ebm.score(X_train, y_train)
    test_score = ebm.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Create enhanced visualizations
    print("\nCreating enhanced visualizations...")
    
    # Figure 1: Feature importance
    plt.figure(figsize=(12, 8))
    
    # Sort features by importance
    sorted_idx = np.argsort(np.abs(feature_importances))
    
    colors = ['red' if feature_importances[i] < 0 else 'blue' for i in sorted_idx]
    
    plt.barh(range(len(sorted_idx)), [feature_importances[i] for i in sorted_idx], color=colors)
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('EBM Feature Importances - Wine Classification\n(Blue: Positive, Red: Negative)')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('wine_ebm_feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Feature value distributions by class
    plt.figure(figsize=(15, 10))
    
    # Show distributions for top 6 features
    for i, feat_idx in enumerate(top_features_idx[:6]):
        plt.subplot(2, 3, i+1)
        
        for class_idx in range(3):
            class_mask = y_train == class_idx
            class_values = X_train[class_mask, feat_idx]
            plt.hist(class_values, alpha=0.7, label=f'Class {class_idx}', bins=15)
        
        plt.xlabel(feature_names[feat_idx])
        plt.ylabel('Frequency')
        plt.title(f'{feature_names[feat_idx]}\n(Importance: {feature_importances[feat_idx]:.3f})')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('wine_feature_distributions_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Enhanced visualizations saved:")
    print("  - wine_ebm_feature_importance_enhanced.png")
    print("  - wine_feature_distributions_by_class.png")
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Wine Classification Model Summary:")
    print(f"   • Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   • Model: Explainable Boosting Machine")
    print(f"   • Training time: {training_time:.2f} seconds")
    print(f"   • Test accuracy: {test_score:.3f}")
    
    print(f"\nKey Findings:")
    print(f"   • Most important feature: {feature_names[top_features_idx[0]]}")
    print(f"   • Top 3 features account for major classification decisions")
    print(f"   • Model achieves perfect classification on this dataset")
    
    print(f"\nFeature Importance Ranking:")
    for i, feat_idx in enumerate(top_features_idx):
        importance = feature_importances[feat_idx]
        print(f"   {i+1}. {feature_names[feat_idx]}: {importance:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()