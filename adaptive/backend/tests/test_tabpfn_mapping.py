"""
Unit test for TabPFN policy class mapping verification.

Tests that the inference module uses the correct policy class mapping
that matches the training pipeline's LabelEncoder behavior.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.interview.tabpfn_inference import TabPFNInference
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_policy_mapping():
    """Test that policy mapping matches training LabelEncoder."""
    
    logger.info("=" * 60)
    logger.info("POLICY MAPPING UNIT TEST")
    logger.info("=" * 60)
    
    # Expected mapping from training (alphabetical LabelEncoder)
    EXPECTED_MAPPING = {
        0: "Ask Application Question",
        1: "Ask Follow-up Question",
        2: "Increase Difficulty",
        3: "Maintain Difficulty",
        4: "Probe Missing Concept",
        5: "Reduce Difficulty",
        6: "Switch Topic"
    }
    
    try:
        # Initialize TabPFN inference
        tabpfn = TabPFNInference()
        
        # Get actual mapping from inference module
        actual_mapping = tabpfn.VERIFIED_POLICY_MAPPING
        
        logger.info("\n[TEST] Comparing policy mappings:")
        
        # Test each class index
        all_match = True
        for class_idx in range(7):
            expected = EXPECTED_MAPPING[class_idx]
            actual = actual_mapping.get(class_idx)
            
            if expected == actual:
                logger.info(f"  ✓ Class {class_idx}: {actual}")
            else:
                logger.error(f"  ✗ Class {class_idx}: Expected '{expected}', Got '{actual}'")
                all_match = False
        
        if all_match:
            logger.info("\n[TEST] ✓ ALL POLICY MAPPINGS MATCH")
            return True
        else:
            logger.error("\n[TEST] ✗ POLICY MAPPING MISMATCH DETECTED")
            return False
            
    except Exception as e:
        logger.error(f"\n[TEST] ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_ranges():
    """Test that feature ranges match training dataset."""
    
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE RANGES UNIT TEST")
    logger.info("=" * 60)
    
    # Expected ranges from training dataset verification
    EXPECTED_RANGES = {
        "Correctness Score": (3, 100),
        "Concept Coverage": (0, 100),
        "Reasoning Score": (0, 100),
        "Missing Concepts": (0, 8),
        "Engagement Score": (0.0, 1.0),
        "Confidence Score": (0.0, 1.0),
        "Hesitation Score": (0.0, 1.0),
        "Eye Contact Score": (0.0, 1.0),
        "Difficulty": (0, 2),
        "Correct Streak": (0, 5),
        "Wrong Streak": (0, 5)
    }
    
    try:
        # Initialize TabPFN inference
        tabpfn = TabPFNInference()
        
        # Get actual ranges from inference module
        actual_ranges = tabpfn.FEATURE_RANGES
        
        logger.info("\n[TEST] Comparing feature ranges:")
        
        # Test each feature
        all_match = True
        for feature_name, expected_range in EXPECTED_RANGES.items():
            actual_range = actual_ranges.get(feature_name)
            
            if expected_range == actual_range:
                logger.info(f"  ✓ {feature_name}: {actual_range}")
            else:
                logger.error(f"  ✗ {feature_name}: Expected {expected_range}, Got {actual_range}")
                all_match = False
        
        if all_match:
            logger.info("\n[TEST] ✓ ALL FEATURE RANGES MATCH")
            return True
        else:
            logger.error("\n[TEST] ✗ FEATURE RANGE MISMATCH DETECTED")
            return False
            
    except Exception as e:
        logger.error(f"\n[TEST] ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_single_pass_inference():
    """Test that single-pass inference works correctly."""
    
    logger.info("\n" + "=" * 60)
    logger.info("SINGLE-PASS INFERENCE UNIT TEST")
    logger.info("=" * 60)
    
    try:
        # Initialize TabPFN inference
        tabpfn = TabPFNInference()
        
        # Create a valid feature vector
        feature_vector = [75.0, 70.0, 72.0, 2.0, 0.65, 0.62, 0.35, 0.66, 1.0, 1.0, 0.0]
        
        # Validate feature vector
        is_valid, msg = tabpfn.validate_feature_vector(feature_vector)
        if not is_valid:
            logger.error(f"[TEST] ✗ Feature vector validation failed: {msg}")
            return False
        
        logger.info(f"[TEST] Feature vector validated: {msg}")
        
        # Make prediction with probabilities
        result = tabpfn.predict_policy(feature_vector, return_probabilities=True)
        
        logger.info(f"[TEST] Prediction successful:")
        logger.info(f"  Predicted class: {result['predicted_class']}")
        logger.info(f"  Predicted policy: {result['predicted_policy']}")
        logger.info(f"  Probabilities returned: {'probabilities' in result}")
        
        # Verify result structure
        if 'predicted_class' not in result:
            logger.error("[TEST] ✗ Missing 'predicted_class' in result")
            return False
        
        if 'predicted_policy' not in result:
            logger.error("[TEST] ✗ Missing 'predicted_policy' in result")
            return False
        
        if 'probabilities' not in result:
            logger.error("[TEST] ✗ Missing 'probabilities' in result")
            return False
        
        # Verify predicted class is in valid range
        if result['predicted_class'] not in range(7):
            logger.error(f"[TEST] ✗ Invalid predicted class: {result['predicted_class']}")
            return False
        
        # Verify predicted policy is in mapping
        if result['predicted_policy'] not in tabpfn.VERIFIED_POLICY_MAPPING.values():
            logger.error(f"[TEST] ✗ Invalid predicted policy: {result['predicted_policy']}")
            return False
        
        logger.info("\n[TEST] ✓ SINGLE-PASS INFERENCE WORKS CORRECTLY")
        return True
        
    except Exception as e:
        logger.error(f"\n[TEST] ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test1 = test_policy_mapping()
    test2 = test_feature_ranges()
    test3 = test_single_pass_inference()
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Policy Mapping Test: {'PASS' if test1 else 'FAIL'}")
    logger.info(f"Feature Ranges Test: {'PASS' if test2 else 'FAIL'}")
    logger.info(f"Single-Pass Inference Test: {'PASS' if test3 else 'FAIL'}")
    
    all_passed = test1 and test2 and test3
    logger.info(f"\nOverall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    
    sys.exit(0 if all_passed else 1)
