# Chatbot checks

The table on the site is read from `site/chatbot-tests.json`. Each entry has the first answer (`initial_answer`, `initial_supported`), the answer after fixes (`answer`, `supported`) and a `note` on what changed.

Questions:

1. **Data:** Where did the data come from?
2. **Results:** How did the model compare with the baseline?
3. **Outside the guide:** How well did the model work on a dataset it was never tested on?

First run: Q1 true but thin, Q2 correct, Q3 invented ("performed poorly", 18.3 minutes). After the guide and prompt fixes: all three correct, and Q3 now says the guide doesn't cover it.

If you change `site/guide.txt` or the prompt in `site/app.js`, re-run the three questions on the live site and update the JSON.
