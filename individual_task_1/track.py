import cv2 as cv
import numpy as np
import time


def multi_tracking(video_path, output_path_without_extension):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.")
        return

    bbox = cv.selectROI("Choose Object:", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, bbox)
    roi = frame[y:y + h, x:x + w]
    cv.destroyWindow("Choose Object:")

    # -------------------------------------
    # Настройка MedianFlow
    # -------------------------------------
    tracker_mf = cv.legacy.TrackerMedianFlow_create()
    tracker_mf.init(frame, bbox)
    out_mf = cv.VideoWriter(f"{output_path_without_extension}_mf.mp4",
                            cv.VideoWriter_fourcc(*'mp4v'),
                            cap.get(cv.CAP_PROP_FPS),
                            (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    # -------------------------------------
    # Настройка KCF
    # -------------------------------------
    tracker_kcf = cv.TrackerKCF_create()
    tracker_kcf.init(frame, bbox)
    out_kcf = cv.VideoWriter(f"{output_path_without_extension}_kcf.mp4",
                            cv.VideoWriter_fourcc(*'mp4v'),
                            cap.get(cv.CAP_PROP_FPS),
                            (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    
    # -------------------------------------
    # Настройка CSRT
    # -------------------------------------
    tracker_csrt = cv.TrackerCSRT_create()
    tracker_csrt.init(frame, bbox)
    out_csrt = cv.VideoWriter(f"{output_path_without_extension}_csrt.mp4",
                            cv.VideoWriter_fourcc(*'mp4v'),
                            cap.get(cv.CAP_PROP_FPS),
                            (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    start_time = time.time()

    fps_mf = 0
    fps_mf_sum = 0
    fps_csrt = 0
    fps_csrt_sum = 0
    fps_kcf = 0
    fps_kcf_sum = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # -------------------------
        # MedianFlow
        # -------------------------
        mf_start = time.time()
        success_mf, bbox_mf = tracker_mf.update(frame)
        frame_mf = frame.copy()
        if success_mf:
            x_mf, y_mf, w_mf, h_mf = map(int, bbox_mf)
            cv.rectangle(frame_mf, (x_mf, y_mf), (x_mf + w_mf, y_mf + h_mf), (0, 0, 255), 2)

        mf_end = time.time()
        fps_mf = 1.0 / (mf_end - mf_start) if (mf_end - mf_start) > 0 else 0
        fps_mf_sum += fps_mf

        cv.putText(frame_mf, f"Median Flow FPS: {fps_mf:.1f}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out_mf.write(frame_mf)
        cv.imshow("Median Flow", frame_mf)


        # -------------------------
        # KCF
        # -------------------------
        kcf_start = time.time()
        success_k, bbox_k = tracker_kcf.update(frame)
        frame_kcf = frame.copy()
        if success_k:
            x_k, y_k, w_k, h_k = map(int, bbox_k)
            cv.rectangle(frame_kcf, (x_k, y_k), (x_k + w_k, y_k + h_k), (0, 0, 255), 2)

        kcf_end = time.time()
        fps_kcf = 1.0 / (kcf_end - kcf_start) if (kcf_end - kcf_start) > 0 else 0
        fps_kcf_sum += fps_kcf

        cv.putText(frame_kcf, f"KCF FPS: {fps_kcf:.1f}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_kcf.write(frame_kcf)
        cv.imshow("KCF", frame_kcf)

        # -------------------------
        # CSRT
        # -------------------------
        csrt_start = time.time()
        success_c, bbox_c = tracker_csrt.update(frame)
        frame_csrt = frame.copy()
        if success_c:
            x_c, y_c, w_c, h_c = map(int, bbox_c)
            cv.rectangle(frame_csrt, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 0, 255), 2)

        csrt_end = time.time()
        fps_csrt = 1.0 / (csrt_end - csrt_start) if (csrt_end - csrt_start) > 0 else 0
        fps_csrt_sum += fps_csrt

        cv.putText(frame_csrt, f"CSRT FPS: {fps_csrt:.1f}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_csrt.write(frame_csrt)
        cv.imshow("CSRT", frame_csrt)

        

        current_time = time.time() - start_time
        overall_fps = frame_count / current_time if current_time > 0 else 0

        print(
            f"\rКадр: {frame_count} | Median Flow: {(fps_mf_sum / frame_count):.1f} FPS | CSRT: {fps_csrt:.1f} FPS | KCF: {fps_kcf:.1f} FPS | Общий: {overall_fps:.1f} FPS",
            end="", flush=True)

        if cv.waitKey(10) & 0xFF == 27:
            break


    print(f"\n\n=== СТАТИСТИКА ===")
    print(f"Всего кадров: {frame_count}")
    print(f"Средний FPS (MedianFlow): {(fps_mf_sum / frame_count):.1f}")
    print(f"Средний FPS (CSRT): {(fps_csrt_sum / frame_count):.1f}")
    print(f"Средний FPS (KCF): {(fps_kcf_sum / frame_count):.1f}")

    cap.release()
    out_mf.release()
    out_csrt.release()
    out_kcf.release()
    cv.destroyAllWindows()


video_name = "BasketRing"
multi_tracking(f"./videos/input/{video_name}.mp4", f"./videos/output/{video_name}")
