from voice_transcribe.prompts import load_process_prompt


def test_load_process_prompt_injects_transcript():
    transcript = "hello from transcript"
    prompt = load_process_prompt(transcript)

    assert transcript in prompt
    assert "{transcript}" not in prompt
