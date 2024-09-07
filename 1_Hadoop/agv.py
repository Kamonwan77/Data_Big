from mrjob.job import MRJob
# สร้างคลาส MapReduceAvg ที่สืบทอดมาจาก MRJob เพื่อใช้งาน MapReduce
class MapReduceAvg(MRJob):

    # ฟังก์ชัน mapper สำหรับแยกและจัดการข้อมูลจากแต่ละบรรทัด
    def mapper(self, _, line):
        # แยกข้อมูลแต่ละบรรทัดออกเป็นรายการ โดยใช้เครื่องหมายคอมม่า (,) เป็นตัวแบ่ง
        data = line.split(',')
        # ดึงค่า status_type จากคอลัมน์ที่ 2 (index 1)
        status_type = data[1].strip()
        # ดึงค่าจำนวนการตอบสนองจากคอลัมน์ที่ 4 (index 3)
        num_reactions = data[3].strip()
        try:
            # ส่งออกค่าคู่ (status_type, num_reactions) โดยแปลง num_reactions ให้เป็น float
            yield status_type, float(num_reactions)
        except:
            # ถ้าเกิดข้อผิดพลาด (เช่น ไม่สามารถแปลงเป็น float ได้) ให้ข้ามไป
            pass
    # ฟังก์ชัน reducer สำหรับคำนวณค่าเฉลี่ยของแต่ละกลุ่ม status_type
    def reducer(self, key, values):  
        # แปลง values ให้เป็นลิสต์เพื่อนำไปคำนวณ
        lval = list(values) 
        # ส่งออกค่าคู่ (key, ค่าเฉลี่ย) โดยใช้ sum หารด้วย len และปัดเศษให้มีทศนิยม 2 ตำแหน่ง
        yield key, round(sum(lval)/len(lval), 2)
# ตรวจสอบว่าถ้าถูกเรียกจาก command line จะทำการรันคลาส MapReduceAvg
if __name__ == "__main__":
    MapReduceAvg.run()
