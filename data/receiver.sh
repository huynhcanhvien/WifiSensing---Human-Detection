#!/bin/bash

# --- CONFIGURATION ---
PORT=9999
DATA_DIR="Data"
# ---------------------

# Create directories
mkdir -p "$DATA_DIR/empty"
mkdir -p "$DATA_DIR/breathing"
mkdir -p "$DATA_DIR/hold_breath"

while true; do
    clear
    echo "========================================"
    echo "   MACBOOK RECEIVER (STATION)           "
    echo "   Your IP: $(ipconfig getifaddr en0)   "
    echo "========================================"
    echo ""
    echo ">>> WAITING FOR SIGNAL FROM PI..."
    
    TEMP_FILE="temp_$(date +%H%M%S).pcap"
    
    # STEP 1: Wait for Handshake Signal
    # Listen for "SIGNAL_START" message from Pi
    nc -l $PORT > "$TEMP_FILE"
    
    FIRST_LINE=$(head -n 1 "$TEMP_FILE" 2>/dev/null)
    
    # Check if it is the start signal
    if [[ "$FIRST_LINE" == *"SIGNAL_START"* ]]; then
        echo ""
        echo ">>> SIGNAL RECEIVED! Pi is preparing..."
        echo ">>> RECORDING DATA... (Waiting for Pi to stop)"
        rm "$TEMP_FILE"
        
        # STEP 2: Open port again to capture Real Data
        # The script will pause here until Pi stops sending
        nc -l $PORT > "$TEMP_FILE"
        
        # STEP 3: Process the file
        if [ -s "$TEMP_FILE" ]; then
            SIZE=$(du -h "$TEMP_FILE" | cut -f1)
            echo ""
            echo "SUCCESS! Received file size: $SIZE"
            echo "----------------------------------------"
            echo "Where to save this file?"
            echo "1. EMPTY (Empty Room)"
            echo "2. BREATHING (Normal Breathing)"
            echo "3. HOLD_BREATH (Holding Breath)"
            echo "4. DELETE (Trash)"
            echo -n ">> Select (1-4): "
            read type
            
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            case $type in
                1) mv "$TEMP_FILE" "$DATA_DIR/empty/empty_${TIMESTAMP}.pcap"; echo "-> Saved to EMPTY";;
                2) mv "$TEMP_FILE" "$DATA_DIR/breathing/breath_${TIMESTAMP}.pcap"; echo "-> Saved to BREATHING";;
                3) mv "$TEMP_FILE" "$DATA_DIR/hold_breath/hold_${TIMESTAMP}.pcap"; echo "-> Saved to HOLD_BREATH";;
                *) rm "$TEMP_FILE"; echo "-> Deleted.";;
            esac
            
            # STEP 4: Ask to Continue or Quit
            echo ""
            echo "----------------------------------------"
            echo "Do you want to measure again?"
            echo "y. YES (Continue waiting)"
            echo "n. NO (Exit program)"
            echo -n ">> Choice (y/n): "
            read continue_choice
            
            if [[ "$continue_choice" == "n" || "$continue_choice" == "N" ]]; then
                echo "Goodbye!"
                exit 0
            fi
        else
            echo "WARNING: File is empty. Deleted."
            rm "$TEMP_FILE" 2>/dev/null
        fi
    fi
done

