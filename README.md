# Apache Kafka, Reddit Dataset

Proyek ini memanfaatkan **Apache Kafka** untuk memproses data dari **Reddit Dataset**. Data yang dihasilkan digunakan untuk melatih model dengan **Spark**, dan hasilnya dapat diakses melalui API yang disediakan oleh script `api.py`.

## Cara Penggunaan

### 1. Jalankan Kafka dan Zookeeper
Gunakan perintah berikut untuk memulai `zk-single-kafka-single.yml` dengan Docker Compose:

docker-compose -f zk-single-kafka-single.yml up -d

![Screenshot from 2025-01-02 10-22-29](https://github.com/user-attachments/assets/119b631b-56fd-4220-a926-a499941e5464)

Perintah ini akan menjalankan Zookeeper dan Kafka dalam mode latar belakang (detached).

### 2. Jalankan Kafka Producer dan Consumer dengan Python
Gunakan perintah berikut untuk menjalankan Producer dan Consumer:

- **Producer:**


python kafka_producer.py

![Screenshot from 2025-01-02 10-39-40](https://github.com/user-attachments/assets/92c0a706-679b-4037-a3c3-488c1bb37f02)

- **Consumer:**

python kafka_consumer.py

![Screenshot from 2025-01-02 10-40-02](https://github.com/user-attachments/assets/57c41dac-74e2-4de2-8388-02d09fafb2f0)

Maka Akan terbentuk 3 Batch CSV 

![Screenshot from 2025-01-02 10-42-04](https://github.com/user-attachments/assets/2408c502-8042-44ff-999e-108bfef8bc35)

![Screenshot from 2025-01-02 10-42-11](https://github.com/user-attachments/assets/d5ec3d78-3897-4de7-b139-ed3224de4bb4)

Berikut tampilan dari conduktor :

![image](https://github.com/user-attachments/assets/beedf40c-0e3d-409a-9005-c18fe0ae93d4)

### 3. Jalankan Script Spark untuk Melatih Model
Setelah file CSV dihasilkan oleh Kafka Consumer, jalankan script **train_model2.py** untuk melatih model:

python train_model2.py

Model yang dihasilkan akan disimpan di folder `model/` dengan nama:

- cure_initial_kmeans_model
- gaussian_mixture_model
- kmeans_model

### 4. Jalankan API untuk Membaca dan Memproses Data
Gunakan script **api.py** untuk menyediakan API yang dapat membaca data dan mengembalikan hasil berdasarkan permintaan:

python api.py



