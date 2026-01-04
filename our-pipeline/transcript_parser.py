# -*- coding: utf-8 -*-
"""Parser for Stanford therapy transcript format (PATIENT/COUNSELOR turns)."""

import re
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversationTurn:
    """Represents a single turn in a therapy conversation."""
    turn_number: int
    role: str  # 'patient' or 'counselor'
    content: str
    timestamp: str  # Optional timestamp like [2:18]


def parse_transcript_text(transcript_text: str) -> List[ConversationTurn]:
    """Parse raw transcript text into structured conversation turns.
    
    Handles the Stanford transcript format where turns are marked as:
    PATIENT: <content>
    COUNSELOR: <content>
    
    Also handles timestamps in format [MM:SS] or [M:SS].
    
    Args:
        transcript_text: Raw transcript text content
        
    Returns:
        List of ConversationTurn objects in order of appearance
        
    Raises:
        ValueError: If transcript format is invalid or empty
    """
    if not transcript_text or not transcript_text.strip():
        raise ValueError("Transcript text is empty")
    
    # Pattern to match PATIENT: or COUNSELOR: at start of line or after newline
    # Captures the role and everything until the next role marker
    turn_pattern = re.compile(
        r'(PATIENT|COUNSELOR):\s*(.*?)(?=(?:PATIENT|COUNSELOR):|END TRANSCRIPT|$)',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern to extract timestamps like [2:18] or [44:55]
    timestamp_pattern = re.compile(r'\[(\d{1,2}:\d{2})\]')
    
    matches = turn_pattern.findall(transcript_text)
    
    if not matches:
        raise ValueError("No valid PATIENT/COUNSELOR turns found in transcript")
    
    turns: List[ConversationTurn] = []
    turn_number = 0
    
    for role_raw, content_raw in matches:
        role = role_raw.lower().strip()
        content = content_raw.strip()
        
        # Skip empty content
        if not content:
            continue
        
        # Extract timestamp if present
        timestamp_match = timestamp_pattern.search(content)
        timestamp = timestamp_match.group(1) if timestamp_match else ""
        
        # Remove timestamp from content for cleaner text
        content_clean = timestamp_pattern.sub('', content).strip()
        
        # Normalize whitespace
        content_clean = re.sub(r'\s+', ' ', content_clean)
        
        turn_number += 1
        turns.append(ConversationTurn(
            turn_number=turn_number,
            role=role,
            content=content_clean,
            timestamp=timestamp
        ))
    
    return turns


def parse_transcript_file(file_path: str) -> List[ConversationTurn]:
    """Parse a transcript file into structured conversation turns.
    
    Args:
        file_path: Path to transcript file (text format)
        
    Returns:
        List of ConversationTurn objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If transcript format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return parse_transcript_text(text)


def get_counselor_turns(turns: List[ConversationTurn]) -> List[ConversationTurn]:
    """Filter to get only counselor turns from a conversation.
    
    Args:
        turns: List of all conversation turns
        
    Returns:
        List of counselor turns only
    """
    return [turn for turn in turns if turn.role == 'counselor']


def get_patient_turns(turns: List[ConversationTurn]) -> List[ConversationTurn]:
    """Filter to get only patient turns from a conversation.
    
    Args:
        turns: List of all conversation turns
        
    Returns:
        List of patient turns only
    """
    return [turn for turn in turns if turn.role == 'patient']


def turns_to_dict_list(turns: List[ConversationTurn]) -> List[Dict[str, Any]]:
    """Convert ConversationTurn objects to list of dictionaries.
    
    Args:
        turns: List of ConversationTurn objects
        
    Returns:
        List of dictionaries with turn data
    """
    return [
        {
            "turn_number": turn.turn_number,
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp
        }
        for turn in turns
    ]


def get_conversation_context(
    turns: List[ConversationTurn],
    up_to_turn: int,
    max_turns: int = 10
) -> str:
    """Get conversation context up to a specific turn.
    
    Useful for providing context to LLM evaluators.
    
    Args:
        turns: List of all conversation turns
        up_to_turn: Include turns up to this turn number
        max_turns: Maximum number of previous turns to include
        
    Returns:
        Formatted conversation context string
    """
    relevant_turns = [t for t in turns if t.turn_number <= up_to_turn]
    
    # Take only the last max_turns
    if len(relevant_turns) > max_turns:
        relevant_turns = relevant_turns[-max_turns:]
    
    context_lines: List[str] = []
    for turn in relevant_turns:
        role_label = "Patient" if turn.role == "patient" else "Counselor"
        context_lines.append(f"{role_label}: {turn.content}")
    
    return "\n\n".join(context_lines)


# Sample transcript text for testing (from the Stanford PDF)
SAMPLE_TRANSCRIPT = """
BEGIN TRANSCRIPT:

PATIENT: I should go first over there.

COUNSELOR: I had just come back in there.

PATIENT: Oh, okay.

COUNSELOR: So, really, you can... I'll leave the door open. You can come in and...

PATIENT: Okay. Here?

COUNSELOR: Yes. You can come in here. If it's ever more than 30 seconds that I'm here.

PATIENT: All right. Speaking of things, I remember that we have scheduled for next Thursday...

COUNSELOR: Yeah. I wanted to... yeah.

PATIENT: ...but something at the company came up.

COUNSELOR: Something has come up?

PATIENT: Yeah, and if you can reschedule or...

COUNSELOR: We can. One of the things I was thinking after last week is we can actually talk about this I think... these kinds of things we can talk about at the end. If you like, we can do that at the end.

PATIENT: Okay. That doesn't... doesn't matter.

COUNSELOR: Doesn't matter?

PATIENT: No. It doesn't matter. I think actually I would rather talk about it.

COUNSELOR: Would you rather do it at the beginning?

PATIENT: Yeah, at the beginning. Although it's part of the thing that I said, I just think like I'd rather pay you at the beginning and like... because it's probably too long. That's the... my mental...

COUNSELOR: Okay. Thank you.

PATIENT: ...structure.

COUNSELOR: Your mental structure?

PATIENT: Yeah. And I also like the fact that at the end I'm... I'm... I think I'm in a different place? But we can try actually. It doesn't matter. [2:18]

COUNSELOR: We can... we can try?

PATIENT: We can try and talk about it next time at the end.

COUNSELOR: Oh, sure. We can try it in different ways. So, Thursday at 2:00 I think is what we had?

PATIENT: Yeah.

COUNSELOR: That's not good?

PATIENT: No. I have a appointment I can't change...

COUNSELOR: Okay.

PATIENT: ...at 12:00.

COUNSELOR: Okay.

PATIENT: It won't... it won't be over before 3:00 or 4:00 even.

COUNSELOR: Okay.

PATIENT: No, 3:00 it will be over.

COUNSELOR: 3:00 it will be over?

PATIENT: 3:00 or 4:00, yeah.

COUNSELOR: I can't do...

PATIENT: I'll take 3:00.

COUNSELOR: I couldn't do later though.

PATIENT: You couldn't... can you do earlier than 3:00?

COUNSELOR: Unfortunately, no. That day is a little bit of a crazy day for me, but I do have some other... some other times have opened up that I don't usually have. Monday the... what are your Monday's like?

PATIENT: That's something weird. Okay, it's not. Monday morning is fine.

COUNSELOR: Monday I could...

PATIENT: Not in the afternoon, but I can do...

COUNSELOR: I could do 12:00.

PATIENT: 12:00 is okay.

COUNSELOR: That's okay?

PATIENT: Yeah.

COUNSELOR: That's the earliest I think I could do. Let me just make sure. Yeah. I can do that time. Okay. That was easy. So, 12:00 noon on Monday. Yes?

PATIENT: Yes.

COUNSELOR: Yes?

PATIENT: Um hm.

COUNSELOR: Need a pen?

PATIENT: Yeah.

COUNSELOR: Okay.

PATIENT: Thank you.

COUNSELOR: I also had wanted just to let you know a little bit. Last week I had let you now that I'm going to be out for several weeks in January, and I wanted to let you know that part of the reason why I'm not sure when I'm coming back is I'm actually having a medical procedure. So, I'm hoping for the best and that I will... it'll be about three or four weeks. It might be sooner and I'm hoping for that and, of course, I'll... I'll let you know about that. But I also just wanted to let you know that when I come back, I'm going to be laid up in bed for a little bit. So... just so you're not surprised. [4:38]

PATIENT: May... can I ask what is going on?

COUNSELOR: Sure. Sure you can ask. I had a car accident last year and I need to have a surgery on my back. So...

PATIENT: So that's... that's why?

COUNSELOR: So, that's why.

PATIENT: It takes place here in Chicago somewhere?

COUNSELOR: In Chicago, yes. Yeah. But it's going... the first few weeks I won't be able to...

PATIENT: To walk.

COUNSELOR: To walk.

PATIENT: Okay. Well, I hope that...

COUNSELOR: I think it'll be fine.

PATIENT: ...that it'll be fine.

COUNSELOR: Yeah. I hope so too.

PATIENT: All right. Was it a bad accident or...

COUNSELOR: It wasn't a life threatening accident...

PATIENT: Right.

COUNSELOR: ...but it was a bad accident. It was. Yeah. I don't know if you've ever had experience being injured or...

PATIENT: Not. No, not really.

COUNSELOR: Yeah.

PATIENT: I mean, the only time that I had something like that was... that I was afraid of my life, but it wasn't a car accident or anything. It was because I had... I had blacked out when I was a kid.

COUNSELOR: Oh. Those can be very serious.

PATIENT: Yeah, but I'm... I'm fine now. It was a long time ago.

COUNSELOR: How old were you?

PATIENT: I think I was 12 or 13. I don't remember exactly. But I remember the feeling. I remember my mom was there and she was really scared. And I remember thinking, like, "Am I going to die?" But then I woke up and I was fine.

COUNSELOR: That must have been frightening for you.

PATIENT: Yeah, it was. But you know, after that, I kind of just... moved on. I didn't really think about it much.

COUNSELOR: It sounds like you've developed a way of coping with difficult experiences by moving forward quickly. Is that something you notice in other areas of your life?

PATIENT: Yeah, I guess so. I mean, I don't like to dwell on things. My parents always said I was resilient. But sometimes I wonder if I'm just avoiding things, you know?

COUNSELOR: That's an insightful observation. What makes you wonder about that?

PATIENT: Well, sometimes things come back. Like, memories or feelings. And I don't know where they come from. It's like I thought I dealt with something, but then it pops up again.

COUNSELOR: That's very common, actually. Sometimes when we move through difficult experiences quickly, we don't fully process them at the time. And those unprocessed experiences can resurface later. What kind of memories or feelings have been coming back for you?

PATIENT: I don't know... it's hard to talk about.

COUNSELOR: Take your time. There's no pressure. We can explore this at whatever pace feels comfortable for you.

PATIENT: Okay. I think... I think it has to do with when I was younger. In high school. There were some... some difficult times.

COUNSELOR: High school can be a challenging time for many people. What was difficult for you during that period?

PATIENT: I was... I was bullied. Pretty badly. And I never really told anyone about it.

COUNSELOR: I'm sorry you went through that. Bullying can have lasting effects. How do you feel about sharing that with me now?

PATIENT: Scared, I guess. But also... relieved? Like I've been carrying this around for so long.

COUNSELOR: Both of those feelings make complete sense. It takes courage to share something you've held for so long. What would it mean for you to explore this further in our sessions?

PATIENT: I think... I think I need to. It's affecting my relationships now. At work, with friends. I get really defensive sometimes, and I don't know why.

COUNSELOR: That's a really important connection you're making - linking past experiences to present behaviors. What do you notice about those moments when you become defensive?

END TRANSCRIPT
"""


if __name__ == "__main__":
    # Test the parser
    turns = parse_transcript_text(SAMPLE_TRANSCRIPT)
    print(f"Parsed {len(turns)} turns")
    
    counselor_turns = get_counselor_turns(turns)
    print(f"Counselor turns: {len(counselor_turns)}")
    
    patient_turns = get_patient_turns(turns)
    print(f"Patient turns: {len(patient_turns)}")
    
    print("\nFirst 5 turns:")
    for turn in turns[:5]:
        print(f"  [{turn.turn_number}] {turn.role.upper()}: {turn.content[:50]}...")

