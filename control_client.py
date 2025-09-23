#!/usr/bin/env python3
"""
control_client.py - Control Client for Go2 Robot Simulation
Run this to send commands to the simulation server
"""

import socket
import json
import sys
import time
import numpy as np
from typing import Dict, Any, Optional


class Go2ControlClient:
    def __init__(self, host: str = 'localhost', port: int = 5555):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to simulation server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Receive initial connection message
            response = self.receive_response()
            if response:
                print(f"[CLIENT] {response.get('message', 'Connected')}")
            return True
        except Exception as e:
            print(f"[CLIENT] Failed to connect: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("[CLIENT] Disconnected")
            
    def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to server"""
        if not self.connected:
            print("[CLIENT] Not connected to server")
            return False
            
        try:
            msg = json.dumps(command).encode('utf-8')
            self.socket.send(msg)
            return True
        except Exception as e:
            print(f"[CLIENT] Error sending command: {e}")
            return False
            
    def receive_response(self, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """Receive response from server"""
        if not self.connected:
            return None
            
        try:
            self.socket.settimeout(timeout)
            data = self.socket.recv(1024)
            if data:
                return json.loads(data.decode('utf-8'))
        except socket.timeout:
            print("[CLIENT] Response timeout")
        except Exception as e:
            print(f"[CLIENT] Error receiving response: {e}")
        return None
        
    def send_and_receive(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send command and wait for response"""
        if self.send_command(command):
            return self.receive_response()
        return None
        
    # High-level command methods
    def stand(self, speed: float = 0.5):
        """Make robot stand"""
        response = self.send_and_receive({'type': 'stand', 'speed': speed})
        if response:
            print(f"[RESPONSE] {response.get('message', 'No message')}")
            
    def sit(self, speed: float = 0.5):
        """Make robot sit"""
        response = self.send_and_receive({'type': 'sit', 'speed': speed})
        if response:
            print(f"[RESPONSE] {response.get('message', 'No message')}")
            
    def move_to_position(self, position: str, speed: float = 0.5):
        """Move to predefined position"""
        response = self.send_and_receive({'type': 'move_to', 'position': position, 'speed': speed})
        if response:
            print(f"[RESPONSE] {response.get('message', 'No message')}")
            
    def custom_position(self, joints: list, speed: float = 0.5):
        """Move to custom joint positions"""
        response = self.send_and_receive({'type': 'custom_position', 'joints': joints, 'speed': speed})
        if response:
            print(f"[RESPONSE] {response.get('message', 'No message')}")
            
    def stop(self):
        """Stop current motion"""
        response = self.send_and_receive({'type': 'stop'})
        if response:
            print(f"[RESPONSE] {response.get('message', 'No message')}")
            
    def reset(self):
        """Reset simulation"""
        response = self.send_and_receive({'type': 'reset'})
        if response:
            print(f"[RESPONSE] {response.get('message', 'No message')}")
            
    def get_state(self):
        """Get robot state"""
        response = self.send_and_receive({'type': 'get_state'})
        if response and response.get('status') == 'success':
            state = response.get('state', {})
            print("\n[ROBOT STATE]")
            print(f"  State: {state.get('robot_state', 'unknown')}")
            print(f"  Controller Active: {state.get('controller_active', False)}")
            print(f"  Trajectory Progress: {state.get('trajectory_progress', 0):.1%}")
            
            joints = state.get('joint_positions', [])
            if joints:
                print(f"  Joint Positions: {[f'{j:.2f}' for j in joints[:4]]} ...")
                
    def shutdown_server(self):
        """Shutdown the simulation server"""
        response = self.send_and_receive({'type': 'shutdown'})
        if response:
            print(f"[RESPONSE] {response.get('message', 'No message')}")


def interactive_mode(client: Go2ControlClient):
    """Interactive command-line interface"""
    print("\n" + "="*50)
    print("Go2 Robot Control Interface")
    print("="*50)
    print("\nAvailable commands:")
    print("  stand [speed]     - Make robot stand (speed: 0.1-2.0)")
    print("  sit [speed]       - Make robot sit")
    print("  neutral           - Move to neutral position")
    print("  stretch           - Stretch pose")
    print("  stop              - Stop current motion")
    print("  reset             - Reset simulation")
    print("  state             - Get robot state")
    print("  custom <j1> <j2> ... <j12> - Set custom joint angles")
    print("  demo              - Run demo sequence")
    print("  exit/quit         - Exit client")
    print("  shutdown          - Shutdown server and exit")
    print("\n")
    
    while True:
        try:
            cmd_input = input("Command> ").strip().lower().split()
            
            if not cmd_input:
                continue
                
            cmd = cmd_input[0]
            
            if cmd in ['exit', 'quit']:
                print("Exiting...")
                break
                
            elif cmd == 'stand':
                speed = float(cmd_input[1]) if len(cmd_input) > 1 else 0.5
                client.stand(speed)
                
            elif cmd == 'sit':
                speed = float(cmd_input[1]) if len(cmd_input) > 1 else 0.5
                client.sit(speed)
                
            elif cmd == 'neutral':
                client.move_to_position('neutral')
                
            elif cmd == 'stretch':
                client.move_to_position('stretch')
                
            elif cmd == 'stop':
                client.stop()
                
            elif cmd == 'reset':
                client.reset()
                
            elif cmd == 'state':
                client.get_state()
                
            elif cmd == 'custom':
                if len(cmd_input) == 13:  # cmd + 12 joint values
                    joints = [float(x) for x in cmd_input[1:]]
                    client.custom_position(joints)
                else:
                    print("Error: Need 12 joint values")
                    
            elif cmd == 'demo':
                run_demo_sequence(client)
                
            elif cmd == 'shutdown':
                client.shutdown_server()
                print("Server shutdown requested")
                break
                
            else:
                print(f"Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"Error: {e}")


def run_demo_sequence(client: Go2ControlClient):
    """Run a demonstration sequence"""
    print("\n[DEMO] Starting demo sequence...")
    
    sequences = [
        ("Sitting down...", lambda: client.sit(0.3)),
        ("Standing up...", lambda: client.stand(0.5)),
        ("Moving to stretch position...", lambda: client.move_to_position('stretch', 0.4)),
        ("Back to standing...", lambda: client.stand(0.4)),
        ("Moving to neutral...", lambda: client.move_to_position('neutral', 0.3)),
        ("Final stand position...", lambda: client.stand(0.5)),
    ]
    
    for msg, action in sequences:
        print(f"[DEMO] {msg}")
        action()
        time.sleep(3)  # Wait for action to complete
        
    print("[DEMO] Demo sequence complete!")


def batch_mode(client: Go2ControlClient, commands: list):
    """Execute a batch of commands"""
    for cmd in commands:
        print(f"Executing: {cmd}")
        
        if cmd == 'stand':
            client.stand()
        elif cmd == 'sit':
            client.sit()
        elif cmd == 'reset':
            client.reset()
        elif cmd == 'demo':
            run_demo_sequence(client)
        else:
            print(f"Unknown batch command: {cmd}")
            
        time.sleep(2)  # Delay between commands


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Go2 Robot Control Client')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5555, help='Server port (default: 5555)')
    parser.add_argument('--batch', nargs='+', help='Execute batch commands')
    parser.add_argument('--demo', action='store_true', help='Run demo sequence')
    
    args = parser.parse_args()
    
    # Create client
    client = Go2ControlClient(host=args.host, port=args.port)
    
    # Connect to server
    print(f"[CLIENT] Connecting to {args.host}:{args.port}...")
    if not client.connect():
        print("[CLIENT] Failed to connect to server. Is the server running?")
        sys.exit(1)
        
    try:
        if args.demo:
            # Run demo and exit
            run_demo_sequence(client)
        elif args.batch:
            # Execute batch commands
            batch_mode(client, args.batch)
        else:
            # Interactive mode
            interactive_mode(client)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()