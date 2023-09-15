#cli_animations.py
import sys
import time

# Function to show a console loading animation
def loading_animation(event):
    animation_chars = ["|", "/", "-", "\\"]
    i = 0
    sys.stdout.write(' Processing documents: ')
    while not event.is_set():
        if i > 3:
            i = 0
        sys.stdout.write(animation_chars[i] + "\r")
        sys.stdout.flush()
        i += 1
        time.sleep(0.2)
    sys.stdout.write('\nDone!\n')
    sys.stdout.flush()