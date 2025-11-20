# /// script
# description = "Segment images using SAM3 (Segment Anything Model 3)"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13", "transformers", "torch", "pillow"]
# ///

import daft
from daft.functions import decode_image, download
from typing import Any
import torch
from transformers import AutoProcessor, AutoModelForImageSegmentation
from PIL import Image
import io


@daft.cls()
class SAM3Segmentor:
    """
    SAM3 (Segment Anything Model 3) segmentor using daft.cls UDF.
    
    This class loads the SAM3 model from HuggingFace and provides segmentation
    capabilities for images.
    """
    
    def __init__(self, model_id: str = "facebook/sam3", device: str = "cpu"):
        """
        Initialize the SAM3 segmentor.
        
        Args:
            model_id: HuggingFace model ID for SAM3
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
    
    def _load_model(self):
        """Lazy load the model and processor."""
        if self.model is None:
            print(f"Loading SAM3 model: {self.model_id}")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageSegmentation.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
    
    @daft.method(return_dtype=daft.DataType.python())
    def segment_image(self, image_data) -> dict[str, Any]:
        """
        Segment an image using SAM3.
        
        Args:
            image_data: Daft Image data type
            
        Returns:
            Dictionary containing segmentation results with masks and scores
        """
        self._load_model()
        
        # Convert daft Image to PIL Image
        if hasattr(image_data, 'to_pil'):
            pil_image = image_data.to_pil()
        else:
            # Fallback for different daft image representations
            pil_image = Image.fromarray(image_data)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract masks and scores
        # SAM3 outputs contain pred_masks and potentially scores
        pred_masks = outputs.pred_masks.cpu().numpy() if hasattr(outputs, 'pred_masks') else None
        
        result = {
            "num_masks": pred_masks.shape[1] if pred_masks is not None else 0,
            "mask_shape": pred_masks.shape if pred_masks is not None else None,
            "has_masks": pred_masks is not None
        }
        
        return result
    
    @daft.method(return_dtype=daft.DataType.int64())
    def count_segments(self, image_data) -> int:
        """
        Count the number of segments detected in an image.
        
        Args:
            image_data: Daft Image data type
            
        Returns:
            Number of segments detected
        """
        result = self.segment_image(image_data)
        return result.get("num_masks", 0)


if __name__ == "__main__":
    # Create SAM3 segmentor instance
    # Use CPU by default, change to "cuda" if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    segmentor = SAM3Segmentor(model_id="facebook/sam3", device=device)
    
    # Example 1: Using HuggingFace datasets (requires internet access)
    # Uncomment the following to use real images:
    # df = (
    #     daft.from_glob_path("hf://datasets/datasets-examples/doc-image-3/images")
    #     .limit(2)
    #     .with_column("image_bytes", download(daft.col("path")))
    #     .with_column("image", decode_image(daft.col("image_bytes")))
    #     .with_column("image_rgb", daft.col("image").convert_image("RGB").resize(512, 512))
    #     .with_column("segmentation_result", segmentor.segment_image(daft.col("image_rgb")))
    #     .with_column("segment_count", segmentor.count_segments(daft.col("image_rgb")))
    # )
    # df.select("path", "segmentation_result", "segment_count").show()
    
    # Example 2: Using local images (if available)
    # Create synthetic example for demonstration
    import numpy as np
    
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='red')
    
    # Create a dataframe with the test image
    print("\n=== SAM3 Segmentor Demo ===")
    print("This example demonstrates the SAM3 segmentor UDF structure.")
    print("To use with real images, provide image paths using:")
    print("  daft.from_glob_path('path/to/images/*.jpg')")
    print("\nSegmentor initialized successfully!")
    print(f"Model: facebook/sam3")
    print(f"Device: {device}")
    print("\nTo segment images, use:")
    print("  df.with_column('segments', segmentor.segment_image(daft.col('image')))")
    print("  df.with_column('count', segmentor.count_segments(daft.col('image')))")
