import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

class ONNXDetector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
        # Tạo ONNX Runtime session, ưu tiên CUDA
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # In ra các providers khả dụng và đang được sử dụng
        print("Available providers:", ort.get_available_providers())
        print("Using device:", self.session.get_providers())

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Lấy shape của input
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = (self.input_shape[2], self.input_shape[3]) if len(self.input_shape) == 4 else (640, 640)
        
    def preprocess(self, img):
        # Resize và normalize ảnh
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        
        # Chuyển HWC -> CHW (height, width, channels -> channels, height, width)
        img = img.transpose(2, 0, 1)
        
        # Thêm batch dimension
        img = np.expand_dims(img, 0)
        
        return img
    
    def detect(self, img):
        # Lưu kích thước ảnh gốc để scale kết quả về
        original_height, original_width = img.shape[:2]
        
        # Tiền xử lý ảnh
        input_img = self.preprocess(img)
        
        # Chạy inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_img})
        inference_time = time.time() - start_time
        
        # Xử lý output - YOLOv12s có thể có output khác nhau tùy phiên bản
        # Giả sử đầu ra là một tensor có dạng (batch, num_detections, classes+5)
        # với mỗi detection là [x, y, w, h, confidence, class_scores...]
        detections = []
        
        # Cách xử lý phụ thuộc vào định dạng output của model cụ thể
        # Đây là một ví dụ phổ biến cho YOLOv8
        predictions = outputs[0]
        
        for i, pred in enumerate(predictions):
            boxes = pred[:, :4]  # x1, y1, x2, y2 format hoặc xywh tùy model
            scores = pred[:, 4:5] * pred[:, 5:]  # conf * class_prob
            
            # Lấy class scores và class ids
            class_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # Lọc theo ngưỡng confidence
            mask = class_scores >= self.conf_threshold
            boxes, class_scores, class_ids = boxes[mask], class_scores[mask], class_ids[mask]
            
            # Áp dụng NMS
            # (Ví dụ đơn giản - bạn có thể cần điều chỉnh tùy thuộc vào format đầu ra)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), class_scores.tolist(), 
                                      self.conf_threshold, self.iou_threshold)
            
            for idx in indices:
                if isinstance(idx, list) or isinstance(idx, np.ndarray):
                    idx = idx[0]  # Một số phiên bản OpenCV trả về list
                
                # Lấy toạ độ và scales về kích thước ảnh gốc
                x1, y1, x2, y2 = boxes[idx]
                x1 = int(x1 * original_width / self.img_size[1])
                y1 = int(y1 * original_height / self.img_size[0])
                x2 = int(x2 * original_width / self.img_size[1])
                y2 = int(y2 * original_height / self.img_size[0])
                
                # Thêm detection vào kết quả
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(class_scores[idx]),
                    'class_id': int(class_ids[idx])
                })
        
        return detections, inference_time

class PTDetector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        """
        Giả lập detector PT không sử dụng torch trực tiếp
        Trong một ứng dụng thực tế, bạn sẽ cần một cách thức khác để load model PT
        """
        self.model_path = model_path
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
        # Tạo tiến trình riêng để chạy model thông qua Python subprocess
        # Đây là một giả định vì chúng ta không thể sử dụng torch trực tiếp
        print(f"PT model path: {model_path}")
        print("Warning: Không thể load mô hình PT mà không dùng torch")
        print("Để chạy thực tế, bạn cần tạo một API hoặc service riêng để xử lý PT model")
        
    def detect(self, img):
        # Đây là hàm giả lập - trong thực tế bạn cần chạy mô hình PT qua một cách khác
        # Ví dụ tạo một API endpoint riêng chạy PT model
        print("Không thể chạy PT model detection mà không dùng torch")
        return [], 0

class PerformanceEvaluator:
    def __init__(self, onnx_model_path, pt_model_path=None, dataset_path=None, 
                 annotation_path=None, class_names=None):
        self.onnx_detector = ONNXDetector(onnx_model_path)
        
        # Nếu PT model được cung cấp, khởi tạo nó (trong thực tế sẽ cần cách khác)
        self.pt_detector = None
        if pt_model_path:
            print("Cảnh báo: Không thể load PT model mà không dùng torch")
            print("Trong đánh giá thực tế, bạn cần tạo một service riêng cho PT model")
        
        self.dataset_path = dataset_path
        self.annotation_path = annotation_path
        self.class_names = class_names or ['container', 'license_plate']
        
        # Khởi tạo các biến để lưu kết quả
        self.onnx_results = {
            'detections': [],
            'inference_times': [],
            'avg_inference_time': 0,
            'fps': 0
        }
        
        self.pt_results = {
            'detections': [],
            'inference_times': [],
            'avg_inference_time': 0,
            'fps': 0
        }
    
    def load_dataset(self):
        """Load tất cả ảnh từ dataset path"""
        image_files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if self.dataset_path and os.path.exists(self.dataset_path):
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
        
        return image_files
    
    def load_annotations(self):
        """Load annotation data (YOLO format)"""
        annotations = {}
        
        if not self.annotation_path or not os.path.exists(self.annotation_path):
            return annotations
            
        # Nếu annotation_path là thư mục chứa các file .txt theo format YOLO
        if os.path.isdir(self.annotation_path):
            for root, _, files in os.walk(self.annotation_path):
                for file in files:
                    if file.endswith('.txt'):
                        # Tên file không có phần mở rộng
                        img_name = os.path.splitext(file)[0]
                        # Đường dẫn đầy đủ đến file annotation
                        anno_path = os.path.join(root, file)
                        
                        # Tìm đường dẫn ảnh tương ứng
                        img_path = None
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                            candidate = os.path.join(self.dataset_path, img_name + ext)
                            if os.path.exists(candidate):
                                img_path = candidate
                                break
                        
                        if img_path:
                            # Đọc nội dung file annotation
                            gt_boxes = []
                            with open(anno_path, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        # Format YOLO: class_id, x_center, y_center, width, height
                                        class_id = int(parts[0])
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        
                                        # Lưu vào danh sách
                                        gt_boxes.append({
                                            'class_id': class_id,
                                            'bbox': [x_center, y_center, width, height],
                                            'format': 'yolo'  # x_center, y_center, width, height (0-1)
                                        })
                            
                            # Lưu các box ground truth
                            annotations[img_path] = gt_boxes
        
        return annotations
    
    def evaluate_on_images(self, image_files):
        """Đánh giá hiệu suất trên một tập ảnh"""
        self.onnx_results['detections'] = []
        self.onnx_results['inference_times'] = []
        
        # Nếu có PT detector
        if self.pt_detector:
            self.pt_results['detections'] = []
            self.pt_results['inference_times'] = []
        
        print(f"Đánh giá trên {len(image_files)} ảnh...")
        for img_path in tqdm(image_files):
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không thể đọc ảnh: {img_path}")
                continue
            
            # ONNX detection
            onnx_dets, onnx_time = self.onnx_detector.detect(img)
            self.onnx_results['detections'].append({
                'img_path': img_path,
                'detections': onnx_dets
            })
            self.onnx_results['inference_times'].append(onnx_time)
            
            # PT detection (nếu có)
            if self.pt_detector:
                pt_dets, pt_time = self.pt_detector.detect(img)
                self.pt_results['detections'].append({
                    'img_path': img_path,
                    'detections': pt_dets
                })
                self.pt_results['inference_times'].append(pt_time)
        
        # Tính thời gian trung bình và FPS
        if self.onnx_results['inference_times']:
            self.onnx_results['avg_inference_time'] = np.mean(self.onnx_results['inference_times'])
            self.onnx_results['fps'] = 1.0 / self.onnx_results['avg_inference_time'] if self.onnx_results['avg_inference_time'] > 0 else 0
        
        if self.pt_detector and self.pt_results['inference_times']:
            self.pt_results['avg_inference_time'] = np.mean(self.pt_results['inference_times'])
            self.pt_results['fps'] = 1.0 / self.pt_results['avg_inference_time'] if self.pt_results['avg_inference_time'] > 0 else 0
        
        return {
            'onnx': self.onnx_results,
            'pt': self.pt_results if self.pt_detector else None
        }
    
    def calculate_metrics(self, results, annotations, iou_threshold=0.5):
        """Tính các metrics như precision, recall, F1-score, mAP"""
        metrics = {
            'per_class': {},
            'overall': {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'mAP': 0
            }
        }
        
        # Nếu không có annotations, chỉ báo cáo thời gian
        if not annotations:
            return metrics
        
        # Khởi tạo các danh sách cho mỗi class
        for class_id, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'ap': 0,
                'detections': []  # Để tính AP (confidence, correct/incorrect)
            }
        
        # Xử lý từng ảnh
        for img_result in results['detections']:
            img_path = img_result['img_path']
            detections = img_result['detections']
            
            # Bỏ qua nếu không có annotation cho ảnh này
            if img_path not in annotations:
                continue
            
            # Lấy ground truth boxes
            gt_boxes = annotations[img_path]
            gt_used = [False] * len(gt_boxes)  # Đánh dấu các gt box đã được match
            
            # Xử lý từng detection
            for det in detections:
                class_id = det['class_id']
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                confidence = det['confidence']
                bbox = det['bbox']  # Định dạng [x1, y1, x2, y2] (pixels)
                
                # Chuyển về định dạng chuẩn để so sánh
                # Giả sử ground truth trong định dạng YOLO (x_center, y_center, width, height) với giá trị từ 0-1
                # và detection trong định dạng [x1, y1, x2, y2] (pixels)
                
                # Lấy kích thước ảnh để chuyển đổi tọa độ
                img = cv2.imread(img_path)
                img_height, img_width = img.shape[:2]
                
                # Convert detection bbox từ [x1, y1, x2, y2] pixels sang định dạng YOLO [x_center, y_center, width, height] (0-1)
                x1, y1, x2, y2 = bbox
                det_x_center = (x1 + x2) / 2 / img_width
                det_y_center = (y1 + y2) / 2 / img_height
                det_width = (x2 - x1) / img_width
                det_height = (y2 - y1) / img_height
                det_normalized = [det_x_center, det_y_center, det_width, det_height]
                
                is_correct = False
                max_iou = 0
                matched_gt_idx = -1
                
                # So sánh với mỗi ground truth box
                for gt_idx, gt in enumerate(gt_boxes):
                    # Bỏ qua nếu không cùng class
                    if gt['class_id'] != class_id:
                        continue
                    
                    # Bỏ qua nếu gt box này đã được match
                    if gt_used[gt_idx]:
                        continue
                    
                    # Tính IoU
                    gt_bbox = gt['bbox']  # [x_center, y_center, width, height] (0-1)
                    gt_x_center, gt_y_center, gt_width, gt_height = gt_bbox
                    
                    # Tính tọa độ corner box từ format YOLO
                    gt_x1 = (gt_x_center - gt_width/2) * img_width
                    gt_y1 = (gt_y_center - gt_height/2) * img_height
                    gt_x2 = (gt_x_center + gt_width/2) * img_width
                    gt_y2 = (gt_y_center + gt_height/2) * img_height
                    
                    # Convert detection back to corner format for IoU calculation
                    det_x1 = x1
                    det_y1 = y1
                    det_x2 = x2
                    det_y2 = y2
                    
                    # Tính IoU
                    intersection_x1 = max(gt_x1, det_x1)
                    intersection_y1 = max(gt_y1, det_y1)
                    intersection_x2 = min(gt_x2, det_x2)
                    intersection_y2 = min(gt_y2, det_y2)
                    
                    if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                        det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                        union_area = gt_area + det_area - intersection_area
                        iou = intersection_area / union_area
                    else:
                        iou = 0
                    
                    # Cập nhật nếu IoU lớn hơn
                    if iou > max_iou:
                        max_iou = iou
                        matched_gt_idx = gt_idx
                
                # Kiểm tra nếu detection khớp với ground truth
                if max_iou >= iou_threshold:
                    is_correct = True
                    gt_used[matched_gt_idx] = True
                    metrics['per_class'][class_name]['true_positives'] += 1
                else:
                    metrics['per_class'][class_name]['false_positives'] += 1
                
                # Lưu detection để tính AP
                metrics['per_class'][class_name]['detections'].append({
                    'confidence': confidence,
                    'correct': is_correct
                })
            
            # Đếm false negatives (gt boxes không được match)
            for gt_idx, gt in enumerate(gt_boxes):
                if not gt_used[gt_idx]:
                    class_id = gt['class_id']
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        metrics['per_class'][class_name]['false_negatives'] += 1
        
        # Tính precision, recall, f1-score, AP cho mỗi class
        overall_precision_sum = 0
        overall_recall_sum = 0
        overall_f1_sum = 0
        overall_ap_sum = 0
        valid_class_count = 0
        
        for class_name, class_metrics in metrics['per_class'].items():
            tp = class_metrics['true_positives']
            fp = class_metrics['false_positives']
            fn = class_metrics['false_negatives']
            
            # Precision, recall và F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics['precision'] = precision
            class_metrics['recall'] = recall
            class_metrics['f1_score'] = f1
            
            # Tính AP (Average Precision)
            detections = class_metrics['detections']
            if detections:
                # Sắp xếp theo confidence giảm dần
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Tính precision/recall tại mỗi ngưỡng
                precisions = []
                recalls = []
                correct_count = 0
                
                for i, det in enumerate(detections):
                    if det['correct']:
                        correct_count += 1
                    
                    # Precision và recall tại điểm này
                    current_precision = correct_count / (i + 1)
                    current_recall = correct_count / (tp + fn) if (tp + fn) > 0 else 0
                    
                    precisions.append(current_precision)
                    recalls.append(current_recall)
                
                # Tính AP bằng phương pháp AUC (Area Under Curve)
                if recalls and precisions:
                    ap = 0
                    for i in range(len(recalls) - 1):
                        ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
                    
                    class_metrics['ap'] = ap
                    overall_ap_sum += ap
            
            # Cộng dồn cho overall metrics
            overall_precision_sum += precision
            overall_recall_sum += recall
            overall_f1_sum += f1
            valid_class_count += 1
        
        # Tính overall metrics
        if valid_class_count > 0:
            metrics['overall']['precision'] = overall_precision_sum / valid_class_count
            metrics['overall']['recall'] = overall_recall_sum / valid_class_count
            metrics['overall']['f1_score'] = overall_f1_sum / valid_class_count
            metrics['overall']['mAP'] = overall_ap_sum / valid_class_count
        
        return metrics
    
    def visualize_results(self, img_path, onnx_detections, pt_detections=None, annotations=None, output_path=None):
        """Vẽ kết quả lên ảnh để so sánh"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh: {img_path}")
            return None
        
        img_height, img_width = img.shape[:2]
        
        # Tạo bản sao để vẽ kết quả
        img_onnx = img.copy()
        if pt_detections is not None:
            img_pt = img.copy()
            img_combined = np.hstack([img_onnx, img_pt])
        else:
            img_combined = img_onnx
        
        # Vẽ ONNX detections
        for det in onnx_detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Vẽ bounding box
            cv2.rectangle(img_onnx, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Vẽ label
            label = f"{class_name}: {conf:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(img_onnx, label, (int(x1), int(y1) - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Vẽ PT detections (nếu có)
        if pt_detections is not None:
            for det in pt_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                class_id = det['class_id']
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Vẽ bounding box
                cv2.rectangle(img_pt, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Vẽ label
                label = f"{class_name}: {conf:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(img_pt, label, (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Vẽ ground truth annotations (nếu có)
        if annotations is not None and img_path in annotations:
            gt_boxes = annotations[img_path]
            
            for gt in gt_boxes:
                class_id = gt['class_id']
                bbox = gt['bbox']  # [x_center, y_center, width, height] (0-1)
                
                # Chuyển từ normalized YOLO format sang pixel coordinates
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Vẽ bounding box ground truth lên cả hai hình
                cv2.rectangle(img_onnx, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Label ground truth
                gt_label = f"GT: {class_name}"
                cv2.putText(img_onnx, gt_label, (x1, y1 - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Vẽ lên ảnh PT nếu có
                if pt_detections is not None:
                    cv2.rectangle(img_pt, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img_pt, gt_label, (x1, y1 - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Thêm tiêu đề
        title_height = 40
        title_img = np.ones((title_height, img_combined.shape[1], 3), dtype=np.uint8) * 255
        
        if pt_detections is not None:
            # Tiêu đề cho hình bên trái
            cv2.putText(title_img, "ONNX Detections", (img_width//2 - 80, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Tiêu đề cho hình bên phải
            cv2.putText(title_img, "PT Detections", (img_width + img_width//2 - 70, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            # Chỉ có ONNX
            cv2.putText(title_img, "ONNX Detections", (img_width//2 - 80, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Ghép tiêu đề với hình ảnh
        img_final = np.vstack([title_img, img_combined])
        
        # Lưu ảnh nếu cần
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_final)
        
        return img_final
    
    def generate_report(self, onnx_metrics, pt_metrics=None):
        """Tạo báo cáo so sánh hiệu suất"""
        report = {
            'performance': {
                'onnx': {
                    'avg_inference_time': self.onnx_results['avg_inference_time'],
                    'fps': self.onnx_results['fps']
                }
            },
            'detection': {
                'onnx': onnx_metrics
            },
            'comparison': {}
        }
        
        # Thêm thông tin PT nếu có
        if pt_metrics and self.pt_detector:
            report['performance']['pt'] = {
                'avg_inference_time': self.pt_results['avg_inference_time'],
                'fps': self.pt_results['fps']
            }
            report['detection']['pt'] = pt_metrics
            
            # So sánh tốc độ
            speed_diff = self.pt_results['avg_inference_time'] - self.onnx_results['avg_inference_time']
            speed_ratio = self.pt_results['avg_inference_time'] / self.onnx_results['avg_inference_time'] if self.onnx_results['avg_inference_time'] > 0 else 0
            
            # So sánh accuracy
            acc_diff = pt_metrics['overall']['mAP'] - onnx_metrics['overall']['mAP']
            
            report['comparison'] = {
                'speed_diff': speed_diff,  # Thời gian (giây): PT - ONNX
                'speed_ratio': speed_ratio,  # Tỉ lệ: PT / ONNX
                'onnx_faster_by': f"{(speed_ratio - 1) * 100:.2f}%" if speed_ratio > 1 else f"{(1 - speed_ratio) * 100:.2f}%",
                'accuracy_diff': acc_diff,
                'better_model': 'PT' if acc_diff > 0.05 else 'ONNX' if acc_diff < -0.05 else 'Equivalent'
            }
        
        return report
    
    def visualize_metrics(self, onnx_metrics, pt_metrics=None, output_dir=None):
        """Vẽ biểu đồ so sánh các metrics"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Tạo biểu đồ cho tốc độ
        plt.figure(figsize=(12, 6))
        models = ['ONNX']
        times = [self.onnx_results['avg_inference_time'] * 1000]  # Đổi sang ms
        fps_values = [self.onnx_results['fps']]
        
        if pt_metrics and self.pt_detector:
            models.append('PT')
            times.append(self.pt_results['avg_inference_time'] * 1000)
            fps_values.append(self.pt_results['fps'])
        
        # Biểu đồ thời gian inference
        plt.subplot(1, 2, 1)
        plt.bar(models, times, color=['skyblue', 'lightgreen'][:len(models)])
        plt.title('Inference Time Comparison')
        plt.ylabel('Time (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(times):
            plt.text(i, v + 0.5, f"{v:.2f}ms", ha='center')
        
        # Biểu đồ FPS
        plt.subplot(1, 2, 2)
        plt.bar(models, fps_values, color=['skyblue', 'lightgreen'][:len(models)])
        plt.title('FPS Comparison')
        plt.ylabel('Frames Per Second')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(fps_values):
            plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'speed_comparison.png'))
        
        # Biểu đồ precision, recall và F1-score
        plt.figure(figsize=(15, 6))
        
        # Tạo dữ liệu cho biểu đồ
        class_names = list(onnx_metrics['per_class'].keys())
        onnx_precisions = [onnx_metrics['per_class'][cls]['precision'] for cls in class_names]
        onnx_recalls = [onnx_metrics['per_class'][cls]['recall'] for cls in class_names]
        onnx_f1s = [onnx_metrics['per_class'][cls]['f1_score'] for cls in class_names]
        
        bar_width = 0.2
        index = np.arange(len(class_names))
        
        # Precision
        plt.subplot(1, 3, 1)
        plt.bar(index, onnx_precisions, bar_width, label='ONNX', color='skyblue')
        
        if pt_metrics:
            pt_precisions = [pt_metrics['per_class'][cls]['precision'] for cls in class_names]
            plt.bar(index + bar_width, pt_precisions, bar_width, label='PT', color='lightgreen')
        
        plt.xlabel('Classes')
        plt.ylabel('Precision')
        plt.title('Precision by Class')
        plt.xticks(index + bar_width/2, class_names)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Recall
        plt.subplot(1, 3, 2)
        plt.bar(index, onnx_recalls, bar_width, label='ONNX', color='skyblue')
        
        if pt_metrics:
            pt_recalls = [pt_metrics['per_class'][cls]['recall'] for cls in class_names]
            plt.bar(index + bar_width, pt_recalls, bar_width, label='PT', color='lightgreen')
        
        plt.xlabel('Classes')
        plt.ylabel('Recall')
        plt.title('Recall by Class')
        plt.xticks(index + bar_width/2, class_names)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # F1-Score
        plt.subplot(1, 3, 3)
        plt.bar(index, onnx_f1s, bar_width, label='ONNX', color='skyblue')
        
        if pt_metrics:
            pt_f1s = [pt_metrics['per_class'][cls]['f1_score'] for cls in class_names]
            plt.bar(index + bar_width, pt_f1s, bar_width, label='PT', color='lightgreen')
        
        plt.xlabel('Classes')
        plt.ylabel('F1-Score')
        plt.title('F1-Score by Class')
        plt.xticks(index + bar_width/2, class_names)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
        
        # Biểu đồ mAP
        plt.figure(figsize=(10, 6))
        
        # mAP cho từng class và overall
        class_names_with_overall = class_names + ['Overall']
        onnx_aps = [onnx_metrics['per_class'][cls]['ap'] for cls in class_names] + [onnx_metrics['overall']['mAP']]
        
        plt.bar(index, onnx_aps[:-1], bar_width, label='ONNX', color='skyblue')
        
        if pt_metrics:
            pt_aps = [pt_metrics['per_class'][cls]['ap'] for cls in class_names] + [pt_metrics['overall']['mAP']]
            plt.bar(index + bar_width, pt_aps[:-1], bar_width, label='PT', color='lightgreen')
        
        plt.xlabel('Classes')
        plt.ylabel('Average Precision (AP)')
        plt.title('AP by Class')
        plt.xticks(index + bar_width/2, class_names)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Overall mAP
        plt.figure(figsize=(8, 6))
        overall_models = ['ONNX']
        overall_maps = [onnx_metrics['overall']['mAP']]
        
        if pt_metrics:
            overall_models.append('PT')
            overall_maps.append(pt_metrics['overall']['mAP'])
        
        plt.bar(overall_models, overall_maps, color=['skyblue', 'lightgreen'][:len(overall_models)])
        plt.title('Overall mAP Comparison')
        plt.ylabel('mean Average Precision (mAP)')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(overall_maps):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'map_comparison.png'))
        
        plt.close('all')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv12s Model Performance Comparison')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX model file')
    parser.add_argument('--pt_model', type=str, help='Path to PT model file (optional)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--annotations', type=str, help='Path to annotation directory')
    parser.add_argument('--class_names', type=str, default='container,license_plate', help='Comma-separated class names')
    parser.add_argument('--output_dir', type=str, default='./comparison_results', help='Output directory for results')
    parser.add_argument('--sample_images', type=int, default=5, help='Number of sample images to visualize')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold')
    
    args = parser.parse_args()
    
    # Khởi tạo class_names
    class_names = args.class_names.split(',')
    
    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Comparing models:")
    print(f"ONNX model: {args.onnx_model}")
    if args.pt_model:
        print(f"PT model: {args.pt_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Annotations: {args.annotations if args.annotations else 'Not provided'}")
    print(f"Classes: {class_names}")
    
    # Khởi tạo evaluator
    evaluator = PerformanceEvaluator(
        onnx_model_path=args.onnx_model,
        pt_model_path=args.pt_model,
        dataset_path=args.dataset,
        annotation_path=args.annotations,
        class_names=class_names
    )
    
    # Tải dataset và annotations
    image_files = evaluator.load_dataset()
    annotations = evaluator.load_annotations()
    
    print(f"Loaded {len(image_files)} images and {len(annotations)} annotations")
    
    # Đánh giá trên tập ảnh
    results = evaluator.evaluate_on_images(image_files)
    
    # Tính metrics cho ONNX
    onnx_metrics = evaluator.calculate_metrics(
        results['onnx'], 
        annotations, 
        iou_threshold=args.iou_threshold
    )
    
    # Tính metrics cho PT nếu có
    pt_metrics = None
    if args.pt_model and results['pt']:
        pt_metrics = evaluator.calculate_metrics(
            results['pt'], 
            annotations, 
            iou_threshold=args.iou_threshold
        )
    
    # Tạo báo cáo
    report = evaluator.generate_report(onnx_metrics, pt_metrics)
    
    # Lưu báo cáo JSON
    with open(os.path.join(args.output_dir, 'comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Vẽ biểu đồ metrics
    evaluator.visualize_metrics(
        onnx_metrics, 
        pt_metrics, 
        output_dir=os.path.join(args.output_dir, 'charts')
    )
    
    # Vẽ visualizations cho một số ảnh mẫu
    sample_dir = os.path.join(args.output_dir, 'sample_detections')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Lấy một số ảnh ngẫu nhiên để visualize
    import random
    sample_count = min(args.sample_images, len(image_files))
    sample_images = random.sample(image_files, sample_count)
    
    for idx, img_path in enumerate(sample_images):
        # Tìm ONNX detections cho ảnh này
        onnx_dets = None
        for result in results['onnx']['detections']:
            if result['img_path'] == img_path:
                onnx_dets = result['detections']
                break
        
        # Tìm PT detections cho ảnh này (nếu có)
        pt_dets = None
        if args.pt_model and results['pt']:
            for result in results['pt']['detections']:
                if result['img_path'] == img_path:
                    pt_dets = result['detections']
                    break
        
        # Visualize
        output_path = os.path.join(sample_dir, f"sample_{idx+1}.jpg")
        evaluator.visualize_results(
            img_path, 
            onnx_dets, 
            pt_dets, 
            annotations, 
            output_path=output_path
        )
    
    # Hiển thị báo cáo tóm tắt
    print("\n===== Performance Summary =====")
    print(f"ONNX Model:")
    print(f"  Average Inference Time: {report['performance']['onnx']['avg_inference_time']*1000:.2f} ms")
    print(f"  FPS: {report['performance']['onnx']['fps']:.2f}")
    print(f"  mAP: {onnx_metrics['overall']['mAP']:.4f}")
    print(f"  Precision: {onnx_metrics['overall']['precision']:.4f}")
    print(f"  Recall: {onnx_metrics['overall']['recall']:.4f}")
    print(f"  F1-Score: {onnx_metrics['overall']['f1_score']:.4f}")
    
    if args.pt_model and pt_metrics:
        print(f"\nPT Model:")
        print(f"  Average Inference Time: {report['performance']['pt']['avg_inference_time']*1000:.2f} ms")
        print(f"  FPS: {report['performance']['pt']['fps']:.2f}")
        print(f"  mAP: {pt_metrics['overall']['mAP']:.4f}")
        print(f"  Precision: {pt_metrics['overall']['precision']:.4f}")
        print(f"  Recall: {pt_metrics['overall']['recall']:.4f}")
        print(f"  F1-Score: {pt_metrics['overall']['f1_score']:.4f}")
        
        print(f"\nComparison:")
        print(f"  Speed Difference: {report['comparison']['speed_diff']*1000:.2f} ms")
        faster_model = "ONNX" if report['comparison']['speed_ratio'] > 1 else "PT"
        print(f"  {faster_model} is faster by {report['comparison']['onnx_faster_by']}")
        print(f"  Accuracy Difference (mAP): {report['comparison']['accuracy_diff']:.4f}")
        print(f"  Overall Better Model: {report['comparison']['better_model']}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()