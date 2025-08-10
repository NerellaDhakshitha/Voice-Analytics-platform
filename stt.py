"""
Clean Speech-to-Text Module
Simplified version without logging clutter
"""

import speech_recognition as sr
import time
from typing import Optional

class SpeechToText:
    def __init__(self):
        """Initialize speech recognition with clean output"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Configure recognition settings
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
        
        # Available recognition engines
        self.engines = ['google', 'sphinx']
        
        # Calibrate microphone
        self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                print("🔧 Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Microphone calibrated")
        except Exception as e:
            print(f"⚠️ Microphone calibration warning: {str(e)}")
    
    def recognize_speech(self, timeout: float = 10, phrase_duration: float = 3) -> Optional[str]:
        """
        Recognize speech from microphone with clean output
        
        Args:
            timeout: Maximum time to wait for speech
            phrase_duration: Maximum duration for a single phrase
            
        Returns:
            Recognized text or None if recognition failed
        """
        try:
            # Listen for audio
            print("🎙️  Listening...")
            audio = self._listen_for_audio(timeout, phrase_duration)
            
            if not audio:
                return None
            
            print("🔄 Processing speech...")
            
            # Try recognition with fallback engines
            result = self._recognize_with_fallback(audio)
            
            if result:
                print(f"✅ Recognition successful: '{result}'")
                return result.lower().strip()
            else:
                print("❌ Could not understand speech")
                return None
                
        except Exception as e:
            print(f"❌ Speech recognition error: {str(e)}")
            return None
    
    def _listen_for_audio(self, timeout: float, phrase_duration: float):
        """Listen for audio input"""
        try:
            with self.microphone as source:
                # Adjust for ambient noise periodically
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                
                # Listen for the phrase
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_duration
                )
                return audio
                
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout - no speech detected")
            return None
        except Exception as e:
            print(f"❌ Audio capture error: {str(e)}")
            return None
    
    def _recognize_with_fallback(self, audio) -> Optional[str]:
        """Try multiple recognition engines"""
        
        # Try Google first (most accurate)
        try:
            result = self.recognizer.recognize_google(audio, language='en-US')
            if result:
                return result
        except sr.UnknownValueError:
            print("🔄 Google couldn't understand audio, trying alternatives...")
        except sr.RequestError as e:
            print(f"🔄 Google service error: {e}, trying alternatives...")
        except Exception as e:
            print(f"🔄 Google recognition failed: {e}, trying alternatives...")
        
        # Try Sphinx as fallback (offline)
        try:
            result = self.recognizer.recognize_sphinx(audio)
            if result:
                print("✅ Offline recognition successful")
                return result
        except sr.UnknownValueError:
            print("❌ Offline recognition couldn't understand audio")
        except sr.RequestError as e:
            print(f"⚠️ Offline recognition unavailable: {e}")
        except Exception:
            pass  # Sphinx not available
        
        return None
    
    def test_microphone(self) -> bool:
        """Test if microphone is working"""
        try:
            print("🧪 Testing microphone... Say something!")
            
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            # Try to recognize
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Microphone test successful! Heard: '{text}'")
                return True
            except:
                print("✅ Microphone is capturing audio (recognition may vary)")
                return True
                
        except Exception as e:
            print(f"❌ Microphone test failed: {str(e)}")
            return False
    
    def get_microphone_list(self):
        """List available microphones"""
        print("🎤 Available microphones:")
        for i, mic_name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"   {i}: {mic_name}")
    
    def set_microphone(self, device_index: int):
        """Set specific microphone device"""
        try:
            self.microphone = sr.Microphone(device_index=device_index)
            print(f"✅ Microphone set to device {device_index}")
            self._calibrate_microphone()
        except Exception as e:
            print(f"❌ Failed to set microphone: {str(e)}")

# Quick test function
def test_speech_recognition():
    """Test the speech recognition system"""
    print("🧪 Testing Speech Recognition System")
    print("=" * 40)
    
    stt = SpeechToText()
    
    # Test microphone
    if not stt.test_microphone():
        print("❌ Microphone test failed. Check your audio setup.")
        return
    
    # Test recognition
    print("\n🎤 Say a test phrase...")
    result = stt.recognize_speech(timeout=5)
    
    if result:
        print(f"🎉 Test successful! You said: '{result}'")
    else:
        print("❌ Test failed - could not recognize speech")
    
    print("✅ Speech recognition test completed")

if __name__ == "__main__":
    test_speech_recognition()