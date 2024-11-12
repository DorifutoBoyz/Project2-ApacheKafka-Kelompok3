from kafka import KafkaConsumer
import csv
import time
import os
import signal
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create a Kafka consumer
consumer = KafkaConsumer(
    'anime_topic_6',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='anime_group',
    value_deserializer=lambda m: m.decode('utf-8')  # Decode message as UTF-8 string (no JSON parsing)
)

batch = []
batch_size = 500000  # Set batch size to 500k
batch_counter = 0  # Initialize batch counter

# Create folder for batches if it doesn't exist
if not os.path.exists('batches'):
    os.makedirs('batches')

def signal_handler(sig, frame):
    logging.info('Gracefully shutting down...')
    if batch:
        save_batch_to_csv(batch, f'batches/batch_remaining_{int(time.time())}.csv')
    consumer.close()
    sys.exit(0)

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

def save_batch_to_csv(batch, filename):
    # Write batch data to CSV
    if batch:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write headers if they exist
            writer.writerow(batch[0].split(","))  # Use first message as headers
            writer.writerows([row.split(",") for row in batch[1:]])  # Write remaining rows as data
        logging.info(f'Saved batch to {filename}')

try:
    for message in consumer:
        # Add message to batch
        batch.append(message.value)
        
        # Save batch to file when reaching batch size
        if len(batch) >= batch_size:
            batch_file_name = f'batches/batch_{batch_counter + 1}_{int(time.time())}.csv'
            save_batch_to_csv(batch, batch_file_name)
            batch_counter += 1  # Increment the batch counter
            batch = []  # Reset the batch
            
            # Stop after saving 3 batches
            if batch_counter >= 3:
                logging.info('Reached the limit of 3 batches. Exiting...')
                break

except Exception as e:
    logging.error(f'An error occurred: {e}')

finally:
    # Ensure any remaining messages are saved on exit
    if batch:
        save_batch_to_csv(batch, f'batches/batch_remaining_{int(time.time())}.csv')
    consumer.close()

