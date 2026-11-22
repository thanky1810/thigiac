# Ảnh đầu vào

Các notebook mặc định dùng ảnh tổng hợp được tạo bởi `cv_bt2.data.synthetic`, vì vậy project chạy được ngay cả khi thư mục này trống.

Có thể đặt ảnh thật tại đây để mở rộng thí nghiệm. Với Harris/Harris–Laplace, ảnh cần chuyển sang ảnh xám và nên chuẩn hóa về `[0,1]`. Với Region Growing/SRG, ảnh xám hoặc ảnh màu đều được hỗ trợ; `T` của Region Growing phụ thuộc thang cường độ.

Khi dùng ảnh thật, hãy ghi rõ nguồn ảnh, phép tiền xử lý, tham số và lý do chọn ảnh trong notebook/báo cáo.
