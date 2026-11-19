# Chatbot checks

The results table on the site is read from `site/chatbot-tests.json`. Fill in the real answers there after running the live chat:

- `answer`: the chatbot's actual reply
- `supported`: `true` if the reply matches the site and guide, otherwise `false`
- `note`: what was wrong, or what you changed in `site/guide.txt`

Questions:

1. **Data:** Where did the data come from?
2. **Results:** How did the model compare with the baseline?
3. **Outside the guide:** How well did the model work on a dataset it was never tested on?

Expected correct behavior: answers 1 and 2 match the guide (synthetic course CSV; 4.94 versus 44.27 min mean absolute error). Answer 3 should say the guide doesn't cover it.
