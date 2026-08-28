# LLM Coding open-ended responses
by Daniel Felipe Parra, Sophia Aristizabal Florez

In this repository you will be able to find the code and folders needed to run the tool created for the classification of any open-ended data, in particular one coming from experiments. Although it was used in the paper, any researcher can use this for a similar task.

## Abstract

This paper examines whether large language models (LLMs) can reliably code open-ended responses and free-form communication data in experimental economics. We apply three LLMs (GPT, Claude, and Gemini) under alternative prompt designs and temperature settings to data from three published experiments that differ in coding structure, the degree of textual ambiguity, and the role of communication in the experimental design. In a trust game where the coding task targets explicit commitment language, all three models achieve strong agreement with human benchmarks (Cohen's κ > 0.80) and replicate the finding that promise-making predicts cooperative behavior. In thematic coding of explanations for discrepancies between self- and peer-reported AI use, LLMs reliably recover the dominant category but over-identify rare inductively derived categories by a factor of three to five. In multi-label coding of chat communication in a coordination experiment, agreement is weaker and downstream regressions do not replicate consistently, with some coefficients reversing sign. Model choice and temperature are not first-order determinants of performance. The primary constraint is the structural fit between the coding scheme and LLM capabilities: LLMs succeed when categories are deductively defined and explicitly marked in the text, and fail when coding relies on pragmatic inference or inductively derived themes.

## Step-by-step

The tool follows six steps, from uploading the input file to downloading the coding results.

[Framework description](flowchart%20(1).pdf)


1. **Upload a CSV input file.** The file must contain, at minimum, the individual textual inputs to be analyzed, with the relevant input column explicitly specified.
2. **Select an action.** Choose between **Assign Categories** (classify each open-ended response using categories you define) or **Create Categories** (generate categories and their definitions from the full set of responses).
3. **Model configuration.** Select the LLM provider (ChatGPT, Gemini, or Claude), the specific model version, and the temperature setting.
4. **Prompt configuration.** Build the prompt using five modular sections: Role, Context, Classification Instructions, Format Specification, and Constraints. Each section can be typed directly or uploaded as a `.txt` file. For the Format section, you can also use the pre-structured JSON option, where you only enter category names and definitions and the tool converts them into valid JSON internally.
5. **Choose a classification strategy.** Zero-shot, few-shot, zero-shot Chain-of-Thought (CoT), or few-shot CoT.
6. **Choose a processing mode.** Line-by-line (per-observation) processing, available for all three providers, or batch processing (asynchronous, ~24h turnaround, roughly 50% cheaper), available for ChatGPT and Claude. If processing is interrupted, the tool saves progress in a helper file so it can resume from the last completed observation.

## Get The Code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/sophiearisti/LLM_Coding_open-ended_responses.git

or [download a zip archive](https://github.com/sophiearisti/LLM_Coding_open-ended_responses/archive/refs/heads/main.zip).

## Create a Virtual Environment

You'll need a working Python environment to run the code.

The recommended way to set up your environment is:

Open your terminal or command prompt, navigate to your project folder, and run:

    python -m venv venv

Creating the folder is not enough; you must "enter" the environment to use its isolated Python and Pip. Use the command corresponding to your Operating System:

To use the isolated Python and Pip, you must activate the environment. Choose the command that matches your Operating System:

| Operating System | Command to Activate |
| :--- | :--- |
| **Windows (Command Prompt)** |     `venv\Scripts\activate` |
| **Windows (PowerShell)** |     `.\venv\Scripts\Activate.ps1`|
| **macOS / Linux** |     `source venv/bin/activate`

> [!TIP]
> Activation is successful if you see `(venv)` appear at the start of your command prompt.


When you are finished working, simply run:

    deactivate


## Dependencies Instalation
After activating your virtual environment, install the required libraries using the following command:

    pip install -r requirements.txt

The following libraries and specific versions will be installed in your virtual environment:

| Library | Version | Description |
| :--- | :--- | :--- |
| `streamlit` | 1.56.0 | Web framework for data apps |
| `pandas` | 3.0.1 | Data manipulation and analysis |
| `langchain-openai` | 1.1.10 | LangChain integration for OpenAI models |
| `google-genai` | 1.64.0 | Google Generative AI Python SDK |
| `tqdm` | 4.67.3 | Progress bar utility |
| `python-dotenv` | 1.2.1 | Environment variable management |
| `anthropic` | 0.96.0 | Anthropic API Python SDK |
| `openai` | 2.24.0 | OpenAI API Python SDK |

## Run the tool

Now that all is set, you can run the tool by entering:

    streamlit run v.py

Once you run the command, Streamlit will automatically open a new tab in your default browser. If it doesn't, you can access the application manually at the following address:

**Local URL:** `http://localhost:8501`

## API key
[Claude API](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve)

[Gemini API](https://ai.google.dev/gemini-api/docs/api-key?utm_source=google&utm_medium=cpc&utm_campaign=Cloud-SS-DR-AIS-FY26-global-gsem-1713578&utm_content=text-ad&utm_term=KW_gemini%20key&gad_source=1&gad_campaignid=23417416052&gbraid=0AAAAACn9t67ubGh6PZBN0Y6CIBQ-bhHbR&gclid=CjwKCAjwt7XQBhBkEiwAtStpp3lshNiXZ3F2UckRWVlRh_Cb2RmRcPpMkjKP1v_RhdbKGmrI3FytChoC3DIQAvD_BwE)

[Chat GPT API](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)

The tool asks for the API key directly in its interface when you select a provider (see step 3 above). You can also store your keys locally in a `.env` file (loaded via `python-dotenv`) using the following variable names:

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Results folder

This folder contains the CSV files with the classified responses generated by the **Assign Categories** step: the original input responses plus the columns for each category assigned by the selected LLM.

## How to use the tool

**To assign categories to existing responses:**
1. Upload your CSV file. The tool shows a preliminary view of the data so you can confirm the correct column was detected.
2. Click **Assign categories**.
3. Select the prompting strategy (zero-shot, few-shot, zero-shot CoT, or few-shot CoT).
4. Select the provider and model, and enter your API key.
5. Fill in the five prompt sections (Role, Context, Classification Task, Format, Constraints), either by writing text or uploading `.txt` files. For Format, you can use the pre-structured JSON option instead of writing raw JSON.
6. Choose how to process the data: line-by-line (results downloadable per temperature as they finish) or batch (submit a job, poll its status, then collect and download results; batches in progress appear under **Batches Pending**).
7. Use **Reset and edit prompt** if you need to modify the prompt or start over. Download all current results first, since this button clears the download options for the previous run.

**To create categories from a set of responses:**
1. Upload your CSV file and click **Create categories** instead.
2. Instead of a Classification Task, define a Create Categories task (for example: identify the major thematic categories that appear across a set of chat messages).
3. The tool returns a table with each generated category and its definition, downloadable as CSV for each temperature setting.

## Contact

For questions or feedback, please contact:

- **Daniel Parra** — [daniel.parra.carreno@gmail.com](mailto:daniel.parra.carreno@gmail.com)

## License

Copyright (C) 2026 Daniel Parra and Sophia Aristizabal

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

## Citation

[You are ethically obliged to cite specialist software used to create research outputs.](https://peerj.com/articles/cs-86/) Please cite the following paper:
```bibtex
@unpublished{parra_aristizabal_2026,
  author  = {Parra, Daniel and Aristizabal, Sophia},
  title   = {Coding open-ended responses with large language models: Evidence from three economic experiments},
  year    = {2026},
  month   = {5},
  note    = {Available at SSRN: \url{https://ssrn.com/abstract=6805398}},
}
```
