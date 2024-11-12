from kafka import KafkaProducer
import csv
import logging
import sys
import signal

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Define the batch size and maximum batch count
batch_size = 500000  # 500k rows per batch
max_batches = 3  # Stop after 3 batches
batch = []
batch_counter = 0  # Initialize batch counter

def signal_handler(sig, frame):
    logging.info('Gracefully shutting down...')
    # Send any remaining messages in the batch
    for message in batch:
        producer.send('anime_topic', message.encode('utf-8'))
    producer.flush()  # Ensure all messages are sent
    producer.close()
    sys.exit(0)

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

# Open the CSV file and read its contents
with open('final_animedataset.csv', 'r') as file:
    reader = csv.reader(file)
    
    # Get headers
    headers = next(reader)  # Read the first row as headers
    headers = [header.strip() for header in headers]
    logging.info("Detected column names: %s", headers)

    for row in reader:
        try:
            # Combine headers and row data as CSV format
            csv_row = ",".join(row)  # Keep it as comma-separated values
            # Append to the batch
            batch.append(csv_row)
            
            # Check if the batch size has been reached
            if len(batch) >= batch_size:
                for message in batch:
                    producer.send('anime_topic', message.encode('utf-8'))
                logging.info("Sent batch %d with %d messages", batch_counter + 1, batch_size)
                batch.clear()  # Clear the batch after sending
                batch_counter += 1  # Increment the batch counter

                # Stop after sending the maximum number of batches
                if batch_counter >= max_batches:
                    logging.info('Reached the limit of %d batches. Exiting...', max_batches)
                    break

        except KeyError as e:
            logging.error("Column not found: %s", e)
            logging.error("Row causing the error: %s", row)
            continue  # Skip rows that cause errors

    # Send any remaining messages in the batch
    for message in batch:
        producer.send('anime_topic', message.encode('utf-8'))

producer.flush()  # Ensure all messages are sent
producer.close()

