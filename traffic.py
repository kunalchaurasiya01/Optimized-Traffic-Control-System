import cv2
import time

# === CONFIGURATION ===
CAMERA_FEEDS = [
    "http://192.168.1.46:8080//video",  # Road 1
    "http://192.168.1.42:8080//video",  # Road 2
    "http://192.168.1.43:8080//video",  # Road 3
    "http://192.168.1.44:8080//video"   # Road 4
]

CAR_CASCADE_PATH = "D:\\Traffic\\cars.xml"

# === LOAD VIDEO STREAMS ===
caps = [cv2.VideoCapture(feed) for feed in CAMERA_FEEDS]
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video stream for Road {i+1}")
        exit()

# === LOAD CASCADE ===
car_cascade = cv2.CascadeClassifier(CAR_CASCADE_PATH)
if car_cascade.empty():
    print("Error: Could not load car cascade.")
    exit()

def count_vehicles(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        return 0

    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    cars = car_cascade.detectMultiScale(
        blurred,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    return len(cars)

def decide_next_green(vehicle_counts, served_directions):
    max_vehicles = -1
    green_index = -1
    for i in range(4):
        if i not in served_directions and vehicle_counts[i] > max_vehicles:
            max_vehicles = vehicle_counts[i]
            green_index = i
    return green_index

# === MAIN LOOP ===
try:
    while True:
        # Initial vehicle counts
        vehicle_counts = [count_vehicles(cap) for cap in caps]
        print("\n--- Initial Vehicle Counts ---")
        for i, count in enumerate(vehicle_counts):
            print(f"Road {i+1}: {count} vehicles")

        # Keep track of directions already served
        served_directions = []

        # Serve all 4 directions one by one based on vehicle count
        for _ in range(4):
            green_index = decide_next_green(vehicle_counts, served_directions)

            if green_index == -1:
                print("All directions served or no vehicles detected.")
                break

            vehicle_count = vehicle_counts[green_index]

            # Decide green light duration
            if vehicle_count < 30:
                green_duration = 30
            elif 30 <= vehicle_count < 60:
                green_duration = 60
            else:
                green_duration = 60  # Even if more than 60 vehicles, max 60 seconds

            print(f"\n🟢 Road {green_index+1}: GREEN for {green_duration} seconds (Vehicle Count: {vehicle_count})")
            for i in range(4):
                if i != green_index:
                    print(f"🔴 Road {i+1}: RED")

            # Green light timing
            start_time = time.time()
            while time.time() - start_time < green_duration:
                time.sleep(5)  # Sleeping in smaller chunks (5s) to stay responsive

            # Mark this direction as served
            served_directions.append(green_index)

            # After green time, update vehicle counts again for remaining roads
            print("\nUpdating vehicle counts for remaining roads...")
            for i in range(4):
                if i not in served_directions:
                    vehicle_counts[i] = count_vehicles(caps[i])
                    print(f"Road {i+1}: {vehicle_counts[i]} vehicles")

        print("\n=== Completed One Full Cycle for All Roads ===\n")
        time.sleep(2)  # Small pause before starting next full cycle

except KeyboardInterrupt:
    print("Program interrupted by user.")

# === CLEANUP ===
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
