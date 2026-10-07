# Báo cáo Day 6: Phát hiện vật cản cho robot/drone bằng lọc mặt đất RANSAC và gom cụm DBSCAN

> Thay **mọi** ô có chữ ĐIỀN nằm trong ngoặc vuông bằng nội dung của bạn, xoá luôn cả dấu ngoặc vuông. Lệnh `python tools/check_submission.py` sẽ báo FAIL nếu còn sót bất kỳ chỗ nào.

- **Họ tên:** Trịnh Xuân Huy
- **MSSV:** 2A202602995
- **Lớp:** AI20K-T4
- **Link repo:** https://github.com/TrinhXuanHuy/TrinhXuanHuy-2A202602995-Track4-Day21
- **Topic:** D — Robot/drone obstacle
- **Dataset:** data/kitti_mini
- **Các frame đã dùng:** 000011, 000019, 000025

> Hãy viết ngắn: mỗi mục từ 3 đến 8 dòng, ưu tiên số liệu và hình ảnh.

## 1. Claim
Trong pipeline phát hiện vật cản bằng hình học cổ điển (RANSAC + Euclidean/DBSCAN clustering), tăng ngưỡng khoảng cách mặt đất `distance_threshold` của RANSAC từ 0.10 m lên 0.30 m làm loại bỏ nhầm 100% các vật cản thấp (< 0.40 m) sát mặt đường coi như mặt đất, trong khi giảm `voxel_size` dưới 0.05 m làm tăng thời gian xử lý lên hơn 4 lần mà không cải thiện số vật cản phát hiện được ở cự ly an toàn (< 15 m).


## 2. Evidence

Bảng hoặc plot số liệu, kèm ảnh/video demo. Ghi rõ đường dẫn file trong `results/`.

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
