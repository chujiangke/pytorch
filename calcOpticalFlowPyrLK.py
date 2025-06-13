import cv2
import numpy as np
from loguru import logger

def calculate_optical_flow(video_input, video_output, frame_rate=30):
    """
    计算视频的光流，并将运动轨迹保存为输出视频。

    :param video_input: 输入视频路径或摄像头ID
    :param video_output: 输出视频文件路径
    :param frame_rate: 视频帧率，默认为30
    """
    # 加载视频文件
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        logger.error(f"Could not open video {video_input}")
        return

    # 获取输入视频的宽度、高度、帧率
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (input_width, input_height)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Input video properties: Size={frame_size}, FPS={input_fps}")

    # 创建视频写入对象，使用MP4编码格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, frame_rate, frame_size)

    # 读取第一帧
    ret, frame1 = cap.read()
    if not ret:
        logger.error("Could not read the first frame.")
        cap.release()
        return

    # 转灰度图并检测初始特征点
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if prev_points is None:
        logger.error("No features detected in the first frame.")
        cap.release()
        return

    logger.info("Feature points detected successfully in the first frame.")

    # 创建一个mask图像，用于绘制轨迹
    mask = np.zeros_like(frame1)

    frame_count = 0  # 计数器用于跟踪处理的帧数
    while True:
        # 读取下一帧
        ret, frame2 = cap.read()
        if not ret:
            logger.info("End of video or error in reading frame.")
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算光流
        next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None)

        if next_points is not None and status is not None:
            # 筛选有效点
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]

            # 绘制轨迹和特征点
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)

            # 更新特征点和前一帧
            prev_points = good_new.reshape(-1, 1, 2)
        else:
            logger.warning("Optical flow calculation failed or no points to track.")
            # continue
            break

        # 将轨迹叠加到当前帧
        img = cv2.add(frame2, mask)

        # 显示结果
        cv2.imshow('Optical Flow with Tracking', img)

        # 写入到输出视频
        out.write(img)

        # 更新前一帧灰度图
        prev_gray = next_gray

        frame_count += 1
        logger.info(f"Processed frame {frame_count}")

        # # 按 'q' 键退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     logger.info("Processing terminated by user.")
        #     break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.success(f"Processing complete. Output saved to {video_output}")


if __name__ == "__main__":
    # 设置输入输出路径
    video_input = "test_move.mp4"      # 输入视频路径
    video_output = "test_move.mp4"    # 输出视频文件路径

    # 调用光流计算函数
    calculate_optical_flow(video_input, video_output)
