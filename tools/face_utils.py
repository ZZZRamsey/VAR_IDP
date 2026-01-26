from io import BytesIO
import random
from PIL import Image
import numpy as np
import cv2
import insightface
import torch
from torchvision import transforms
from torch.cuda.amp import autocast
import os

def face_preserving_resize(img, face_bboxes, target_size=512):
    """
    Resize image while ensuring all faces are preserved in the output.
    
    Args:
        img: PIL Image
        face_bboxes: List of [x1, y1, x2, y2] face coordinates
        target_size: Maximum dimension for resizing
        
    Returns:
        Tuple of (resized image, new_bboxes) or (None, None) if faces can't fit
    """
    
    x1_1, y1_1, x2_1, y2_1 = map(int, face_bboxes[0])
    x1_2, y1_2, x2_2, y2_2 = map(int, face_bboxes[1])
    min_x1 = min(x1_1, x1_2)
    min_y1 = min(y1_1, y1_2)
    max_x2 = max(x2_1, x2_2)
    max_y2 = max(y2_1, y2_2)
    # print("min_x1:", min_x1, "min_y1:", min_y1, "max_x2:", max_x2, "max_y2:", max_y2)
    # if any of them is negative, we cannot resize (Idk why this happens)
    if min_x1 < 0 or min_y1 < 0 or max_x2 < 0 or max_y2 < 0:
        return None, None

    # if face width is longer than the image height, or the face height is longer than the image width, we cannot resize
    face_width = max_x2 - min_x1
    face_height = max_y2 - min_y1
    if face_width > img.height or face_height > img.width:
        return None, None
        
    # Create a copy of face_bboxes for transformation
    new_bboxes = []
    for bbox in face_bboxes:
        new_bboxes.append(list(map(int, bbox)))
    
    # Choose cropping strategy based on image aspect ratio
    if img.width > img.height:
        # We need to crop width to make a square
        square_size = img.height
        
        # Calculate valid horizontal crop range that preserves all faces
        left_max = min_x1  # Leftmost position that includes leftmost face
        right_min = max_x2 - square_size  # Rightmost position that includes rightmost face
        
        if right_min <= left_max:
            # We can find a valid crop window
            start = random.randint(int(right_min), int(left_max)) if right_min < left_max else int(right_min)
            start = max(0, min(start, img.width - square_size))  # Ensure within image bounds
        else:
            # Faces are too far apart for square crop - use center of faces
            face_center = (min_x1 + max_x2) // 2
            start = max(0, min(face_center - (square_size // 2), img.width - square_size))
        
        cropped_img = img.crop((start, 0, start + square_size, square_size))
        
        # Adjust bounding box coordinates based on crop
        for bbox in new_bboxes:
            bbox[0] -= start  # x1 adjustment
            bbox[2] -= start  # x2 adjustment
            # y coordinates remain unchanged
    else:
        # We need to crop height to make a square
        square_size = img.width
        
        # Calculate valid vertical crop range that preserves all faces
        top_max = min_y1  # Topmost position that includes topmost face
        bottom_min = max_y2 - square_size  # Bottommost position that includes bottommost face
        
        if bottom_min <= top_max:
            # We can find a valid crop window
            start = random.randint(int(bottom_min), int(top_max)) if bottom_min < top_max else int(bottom_min)
            start = max(0, min(start, img.height - square_size))  # Ensure within image bounds
        else:
            # Faces are too far apart for square crop - use center of faces
            face_center = (min_y1 + max_y2) // 2
            start = max(0, min(face_center - (square_size // 2), img.height - square_size))
        
        cropped_img = img.crop((0, start, square_size, start + square_size))
        
        # Adjust bounding box coordinates based on crop
        for bbox in new_bboxes:
            bbox[1] -= start  # y1 adjustment
            bbox[3] -= start  # y2 adjustment
            # x coordinates remain unchanged
    
    # Calculate scale factor for resizing from square_size to target_size
    scale_factor = target_size / square_size
    
    # Adjust bounding boxes for the resize operation
    for bbox in new_bboxes:
        bbox[0] = int(bbox[0] * scale_factor)
        bbox[1] = int(bbox[1] * scale_factor)
        bbox[2] = int(bbox[2] * scale_factor)
        bbox[3] = int(bbox[3] * scale_factor)
    
    # Final resize to target size
    resized_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Make sure all coordinates are within bounds (0 to target_size)
    # for bbox in new_bboxes:
    #     bbox[0] = max(0, min(bbox[0], target_size - 1))
    #     bbox[1] = max(0, min(bbox[1], target_size - 1))
    #     bbox[2] = max(1, min(bbox[2], target_size))
    #     bbox[3] = max(1, min(bbox[3], target_size))
    
    return resized_img, new_bboxes

def extract_moref(img, json_data, face_size_restriction=100):
    """
    Extract faces from an image based on bounding boxes in JSON data.
    Makes each face square and resizes to 512x512.
    
    Args:
        img: PIL Image or image data
        json_data: JSON object with 'bboxes' and 'crop' information
        
    Returns:
        List of PIL Images, each 512x512, containing extracted faces
    """
    # Ensure img is a PIL Image
    try:
        if not isinstance(img, Image.Image) and not isinstance(img, torch.Tensor) and not isinstance(img, JpegImageFile):
            img = Image.open(BytesIO(img))
        
        bboxes = json_data['bboxes']
        # crop = json_data['crop']
        # print("len of bboxes:", len(bboxes))
        # Recalculate bounding boxes based on crop info
        # new_bboxes = [recalculate_bbox(bbox, crop) for bbox in bboxes]
        new_bboxes = bboxes
        # any of the face is less than 100 * 100, we ignore this image
        for bbox in new_bboxes:
            x1, y1, x2, y2 = bbox
            if x2 - x1 < face_size_restriction or y2 - y1 < face_size_restriction:
                return []
        # print("len of new_bboxes:", len(new_bboxes))
        faces = []
        for bbox in new_bboxes:
            # print("processing bbox")
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate width and height
            width = x2 - x1
            height = y2 - y1
            
            # Make the bounding box square by expanding the shorter dimension
            if width > height:
                # Height is shorter, expand it
                diff = width - height
                y1 -= diff // 2
                y2 += diff - (diff // 2)  # Handle odd differences
            elif height > width:
                # Width is shorter, expand it
                diff = height - width
                x1 -= diff // 2
                x2 += diff - (diff // 2)  # Handle odd differences
            
            # Ensure coordinates are within image boundaries
            img_width, img_height = img.size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Add modified bbox
            modified_bbox = [x1, y1, x2, y2]
            
            # Extract face region
            face_region = img.crop((x1, y1, x2, y2))
            
            # Resize to 512x512
            face_region = face_region.resize((512, 512), Image.LANCZOS)
            
            faces.append(face_region)
        # print("len of faces:", len(faces))
        return faces, modified_bbox
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

def general_face_preserving_resize(img, face_bboxes, target_size=512):
    """
    Resize image while ensuring all faces are preserved in the output.
    Handles any number of faces (1-5).
    
    Args:
        img: PIL Image
        face_bboxes: List of [x1, y1, x2, y2] face coordinates
        target_size: Maximum dimension for resizing
        
    Returns:
        Tuple of (resized image, new_bboxes) or (None, None) if faces can't fit
    """
    # Find bounding region containing all faces
    if not face_bboxes:
        print("Warning: No face bounding boxes provided.")
        return None, None
        
    min_x1 = min(bbox[0] for bbox in face_bboxes)
    min_y1 = min(bbox[1] for bbox in face_bboxes)
    max_x2 = max(bbox[2] for bbox in face_bboxes)
    max_y2 = max(bbox[3] for bbox in face_bboxes)

    # Check for negative coordinates
    if min_x1 < 0 or min_y1 < 0 or max_x2 < 0 or max_y2 < 0:
        # print("Warning: Negative coordinates found in face bounding boxes.")
        # return None, None
        min_x1 = max(min_x1, 0)
        min_y1 = max(min_y1, 0)

    # Check if faces fit within image
    face_width = max_x2 - min_x1
    face_height = max_y2 - min_y1
    if face_width > img.height or face_height > img.width:
        # print("Warning: Faces are too large for the image dimensions.")
        # return None, None
        # Instead of returning None, we will crop the image to fit the faces
        max_x2 = min(max_x2, img.width)
        max_y2 = min(max_y2, img.height)
        min_x1 = max(min_x1, 0)
        min_y1 = max(min_y1, 0)
    # Create a copy of face_bboxes for transformation
    new_bboxes = []
    for bbox in face_bboxes:
        new_bboxes.append(list(map(int, bbox)))
    
    # Choose cropping strategy based on image aspect ratio
    if img.width > img.height:
        # Crop width to make a square
        square_size = img.height
                
        # # Calculate valid horizontal crop range that preserves all faces
        # left_max = min_x1  # Leftmost position that includes leftmost face
        # right_min = max_x2 - square_size  # Rightmost position that includes rightmost face
        
        # if right_min <= left_max:
        #     # We can find a valid crop window
        #     start = random.randint(int(right_min), int(left_max)) if right_min < left_max else int(right_min)
        #     start = max(0, min(start, img.width - square_size))  # Ensure within image bounds
        # else:
        #     # Faces are too far apart for square crop - use center of faces
        #     face_center = (min_x1 + max_x2) // 2
        #     start = max(0, min(face_center - (square_size // 2), img.width - square_size))

        # Center the crop based on the center of the faces
        face_center = (min_x1 + max_x2) // 2
        start = max(0, min(face_center - (square_size // 2), img.width - square_size))
        
        cropped_img = img.crop((start, 0, start + square_size, square_size))
        
        # Adjust bounding box coordinates
        for bbox in new_bboxes:
            bbox[0] -= start
            bbox[2] -= start
    else:
        # Crop height to make a square
        square_size = img.width

        # # Calculate valid vertical crop range that preserves all faces
        # top_max = min_y1  # Topmost position that includes topmost face
        # bottom_min = max_y2 - square_size  # Bottommost position that includes bottommost face
        
        # if bottom_min <= top_max:
        #     # We can find a valid crop window
        #     start = random.randint(int(bottom_min), int(top_max)) if bottom_min < top_max else int(bottom_min)
        #     start = max(0, min(start, img.height - square_size))  # Ensure within image bounds
        # else:
        #     # Faces are too far apart for square crop - use center of faces
        #     face_center = (min_y1 + max_y2) // 2
        #     start = max(0, min(face_center - (square_size // 2), img.height - square_size))

        # Center the crop based on the center of the faces
        face_center = (min_y1 + max_y2) // 2
        start = max(0, min(face_center - (square_size // 2), img.height - square_size))
        
        cropped_img = img.crop((0, start, square_size, start + square_size))
        
        # Adjust bounding box coordinates
        for bbox in new_bboxes:
            bbox[1] -= start
            bbox[3] -= start
    
    # Calculate scale factor and adjust bounding boxes
    scale_factor = target_size / square_size
    
    for bbox in new_bboxes:
        bbox[0] = int(bbox[0] * scale_factor)
        bbox[1] = int(bbox[1] * scale_factor)
        bbox[2] = int(bbox[2] * scale_factor)
        bbox[3] = int(bbox[3] * scale_factor)
    
    # Final resize to target size
    resized_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Make sure all coordinates are within bounds
    for bbox in new_bboxes:
        bbox[0] = max(0, min(bbox[0], target_size - 1))
        bbox[1] = max(0, min(bbox[1], target_size - 1))
        bbox[2] = max(1, min(bbox[2], target_size))
        bbox[3] = max(1, min(bbox[3], target_size))
    
    return resized_img, new_bboxes


def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

def extract_object(birefnet, image):


    if image.mode != 'RGB':
        image = image.convert('RGB')
    input_images = transforms.ToTensor()(image).unsqueeze(0).to('cuda', dtype=torch.bfloat16)

    # Prediction
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze().float()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    
    # Create a binary mask (0 or 255)
    binary_mask = mask.convert("L")
    
    # Create a new image with black background
    result = Image.new("RGB", image.size, (0, 0, 0))
    
    # Paste the original image onto the black background using the mask
    result.paste(image, (0, 0), binary_mask)
    
    return result, mask


# 判断人脸是否处于图像边缘区域（去除上下分图，左右分图的数据）
def is_face_at_edge(face_bbox, img_width, img_height, margin=0.1):
    x1, y1, x2, y2 = face_bbox
    width, height = x2 - x1, y2 - y1
    margin_pixels = min(width, height) * margin
    
    return (x1 <= margin_pixels or x2 >= img_width - margin_pixels or
            y1 <= margin_pixels or y2 >= img_height - margin_pixels)

class FaceExtractor:
    def __init__(self, gpu_id=0):
        self.model = insightface.app.FaceAnalysis(name = "antelopev2", root="./weights")   # ./models/antelopev2/*.onnx
        self.model.prepare(ctx_id=gpu_id, det_thresh=0.4)

    def is_side_face(self, pose, yaw_threshold=35):
        pitch, yaw, roll = pose
        return abs(yaw) > yaw_threshold if pose is not None else False

    # 逻辑：检测人脸 -> 获取第一个人脸的 bbox -> 调用 extract_moref 抠出 512x512 的人脸图 -> 返回该人脸图和对应的特征向量（embedding）
    def extract(self, image: Image.Image):
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None, None
        res = res[0]
        # print(res.keys())
        bbox = res["bbox"]
        # print("len(bbox)", len(bbox))

        moref, _ = extract_moref(image, {"bboxes": [bbox]}, 1)
        # print("len(moref)", len(moref))
        return moref[0], res["embedding"]

    # 保脸裁剪，并且将bbox调整为正方形
    def locate_bboxes(self, image: Image.Image):
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None
        bboxes = []
        for r in res:
            bbox = r["bbox"]
            bboxes.append(bbox)

        _, new_bboxes_ = general_face_preserving_resize(image, bboxes, 512)

        # ensure the bbox is square
        new_bboxes = []
        for bbox in new_bboxes_:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if w > h:
                diff = w - h
                y1 = max(0, y1 - diff // 2)
                y2 = min(512, y2 + diff // 2 + diff % 2)
            else:
                diff = h - w
                x1 = max(0, x1 - diff // 2)
                x2 = min(512, x2 + diff // 2 + diff % 2)
            new_bboxes.append([x1, y1, x2, y2])

        return new_bboxes
    
    # 提取图中所有检测到的人脸及其特征。
    def extract_refs(self, image: Image.Image):
        """
        Extracts reference faces from the image.
        Returns a list of reference images and their arcface embeddings.
        """
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None, None, None, None, None, None
        ref_imgs = []
        arcface_embeddings = []
        bboxes = []
        modified_bbox_list = [] 
        pose_list = []

        for r in res:
            # side_face = self.is_side_face(r)
            # if side_face:
            #     continue
            bbox = r["bbox"]
            pose = r.get("pose", None)
            pose_list.append(pose)
            bboxes.append(bbox)
            moref, modified_bbox = extract_moref(image, {"bboxes": [bbox]}, 1)
            modified_bbox_list.append(modified_bbox)
            ref_imgs.append(moref[0])
            arcface_embeddings.append(r["embedding"])
        return res, ref_imgs, arcface_embeddings, bboxes, modified_bbox_list, pose_list

def main():
    save_dir = "./vis"
    os.makedirs(save_dir, exist_ok=True)
    face_extractor = FaceExtractor()
    def test_face_extractor(path):
        file_name = os.path.basename(path).split(".")[0]
        test_image = Image.open(path).convert("RGB")
        _, ref_imgs, embeddings, bboxes, modified_bboxes, pose_list = face_extractor.extract_refs(test_image)
        if ref_imgs is None:
            print(f"No faces detected in {path}")
            return
        assert len(ref_imgs) == len(embeddings) == len(bboxes) == len(modified_bboxes) == len(pose_list)
        print(f"Extracted {len(ref_imgs)} faces from {path}")
        print("embedding shape:", embeddings[0].shape)
        for i, ref_img in enumerate(ref_imgs):
            ref_img.save(os.path.join(save_dir, f"{file_name}_ref_img_{i}.jpg"))
        # bboxes = face_extractor.locate_bboxes(test_image)
    
        # visualize
        test_image_np = np.array(test_image)
        # for bbox in bboxes:
        #     x1, y1, x2, y2 = map(int, bbox)
        #     cv2.rectangle(test_image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for bbox in modified_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(test_image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for i, pose in enumerate(pose_list):
            print(f"Face {i} pose:", pose)
            if pose is None:
                continue
            print("is_side_face:", face_extractor.is_side_face(pose))
            if face_extractor.is_side_face(pose):
                bbox = bboxes[i]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(test_image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
        test_image_with_bboxes = Image.fromarray(test_image_np)
        test_image_with_bboxes.save(os.path.join(save_dir, f"{file_name}_with_bboxes.jpg"))

    def test_resize(image_path="./assets/demo.jpg", target_size=512):
        """
            Test the general_face_preserving_resize function.
        """
        print(f"Testing general_face_preserving_resize on {image_path}")
        
        # Load image
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        test_image = Image.open(image_path).convert('RGB')
        print(f"Original image size: {test_image.size}")
        
        # Detect faces
        _, _, _, bboxes, _, _ = face_extractor.extract_refs(test_image)
        
        if not bboxes:
            print("No faces detected in the image.")
            return
        
        print(f"Detected {len(bboxes)} faces: {bboxes}")
        
        # Call general_face_preserving_resize
        resized_img, new_bboxes = general_face_preserving_resize(test_image, bboxes, target_size)
        
        if resized_img is None:
            print("Failed to resize image while preserving faces.")
            return
        
        print(f"Resized image size: {resized_img.size}")
        print(f"New bboxes: {new_bboxes}")
        
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        resized_img.save(os.path.join(save_dir, f"{file_name}_resized.jpg"))
        
        # Visualize bboxes on resized image
        resized_np = np.array(resized_img)
        for bbox in new_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(resized_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        resized_with_bboxes = Image.fromarray(resized_np)
        resized_with_bboxes.save(os.path.join(save_dir, f"{file_name}_resized_with_bboxes.jpg"))
        
        print(f"Resized image saved to {os.path.join(save_dir, f'{file_name}_resized.jpg')}")
        print(f"Resized image with bboxes saved to {os.path.join(save_dir, f'{file_name}_resized_with_bboxes.jpg')}")

    # test_face_extractor("./assets/VCG41N1158298345.jpg")

    test_resize("./assets/VCG211387749073.jpg")


if __name__ == "__main__":
    main()

    