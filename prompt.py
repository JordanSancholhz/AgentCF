from config import ATTRIBUTE_DIMENSIONS

def user_prompt_system_role(user_description):
    return f"You are an Amazon buyer.\n Here is your previous self-introduction, exhibiting your past preferences and dislikes:\n '{user_description}'."
# 1
def user_prompt_template(list_of_item_description, pos_item_title, neg_item_title, system_reason):
    return f"Recently, you considered choosing one item from two candidates. The features of these items are:\n {list_of_item_description}.\n\n After comparing based on your preferences, you chose '{neg_item_title}' and rejected the other. Your explanation was:\n '{system_reason}'. \n\n However, after encountering these items, you realized you prefer '{pos_item_title}' and don't like '{neg_item_title}'.\n This indicates an incorrect choice, and your previous judgment about your preferences was mistaken. Your task now is to update your self-introduction with your new preferences and dislikes. \n Follow these steps: \n 1. Analyze misconceptions in your previous judgment and correct them.\n 2. Identify new preferences from '{pos_item_title}' and dislikes from '{neg_item_title}'. \n 3. Summarize your past preferences, merging them with new insights and removing conflicting parts.\n 4. Update your self-introduction, starting with new preferences, then summarizing past ones, followed by dislikes. \n\n Important notes:\n 1. Your output format should be: 'My updated self-introduction: [Your updated self-introduction here].' \n 2. Keep it under 150 words.  \n 3. Be concise and clear. \n 4. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 5. Your self-introduction should be specific and personalized; avoid generic preferences."
# 2
def user_prompt_template_true(list_of_item_description, pos_item_title, neg_item_title, system_reason):
    return f"Recently, you considered choosing one item from two candidates. The features of these items are:\n {list_of_item_description}.\n\n After comparing based on your preferences, you selected '{pos_item_title}' and rejected the other. Your explanation was:\n '{system_reason}'. \n\n After encountering these items, you found that you really like '{pos_item_title}' and dislike '{neg_item_title}'.\n This indicates you made a correct choice, and your judgment about your preferences was accurate. \n Your task now is to update your self-introduction to reflect your preferences and dislikes from this interaction. \n Please follow these steps: \n 1. Analyze your judgment about your preferences and dislikes from your explanation.\n 2. Identify new preferences based on '{pos_item_title}' and dislikes based on '{neg_item_title}'. \n 3. Summarize your past preferences and dislikes from your previous self-introduction, combining them with new insights while removing conflicting parts.\n 4. Update your self-introduction, starting with your new preferences, then summarizing past ones, followed by your dislikes. \n\n Important notes:\n 1. Your output format should be: 'My updated self-introduction: [Your updated self-introduction here].' \n 2. Keep it under 150 words. \n 3. Be concise and clear. \n 4. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 5. Your self-introduction should be specific and personalized; avoid generic preferences."
# 3
def item_prompt_template(user_description, list_of_item_description, pos_item_title, neg_item_title, system_reason):
    return f"User self-introduction, showing preferences and dislikes: '{user_description}'.\n Recently, the user browsed a shopping site and considered two items:\n {list_of_item_description}.\n\n He chose '{neg_item_title}' and rejected the other, explaining: '{system_reason}'. \n\n However, he prefers '{pos_item_title}' instead, indicating an unsuitable choice due to misleading descriptions. He likes '{pos_item_title}' for its features and dislikes '{neg_item_title}' for undesirable traits. Your task is to update the descriptions of these items. \n Follow these steps:\n 1. Analyze features that led to the poor choice and modify them. \n 2. Examine user preferences and dislikes; explore new features of the preferred item aligning with preferences and opposing dislikes, and do the same for the disliked item, highlighting differences. Your analysis should be thorough. \n 3. Incorporate new features into the previous descriptions, preserving valuable content while being concise.\n\n Important notes: \n 1. Your output should be in the following format: 'The updated description of the first item is: [updated description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each updated description cannot exceed 50 words; be concise and clear. \n 3. In your descriptions, refer to user preferences collectively, avoiding specific individual references, e.g., 'the user with ... preferences/dislikes'.\n 4. The updated description should not contradict the item's inherent characteristics, e.g., do not describe a thriller as having a predictably happy ending. \n 5. The updated description should highlight distinguishing features that differentiate this item from others."
# 4
def item_prompt_template_true(user_description, list_of_item_description, pos_item_title, neg_item_title):
    return f"User self-description, showcasing preferences and dislikes: '{user_description}'.\n Recently, the user browsed a shopping site and considered two items:\n {list_of_item_description}.\n\n The user chose '{pos_item_title}' for its features and rejected '{neg_item_title}' for undesirable traits. Your task is to update the descriptions of these items based on these insights. \n Follow these steps:\n 1. Analyze the user's preferences and dislikes from the self-description. \n 2. Explore the chosen item's features that align with preferences and oppose dislikes, and examine the rejected item's features that align with dislikes and oppose preferences. Highlight the differences thoroughly. \n 3. Incorporate new features into the previous descriptions, preserving key information while being concise.\n\n Important notes: \n 1. Your output should be in the following format: 'The updated description of the first item is: [updated description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each updated description cannot exceed 50 words; be concise and clear! \n 3. In your updated descriptions, refer to preferences collectively, avoiding individual references. For example, say 'the user with ... preferences/dislikes'.\n 4. New features should reflect user preferences, and the updated descriptions must not contradict the inherent characteristics of the items, e.g., do not describe a thriller as having a predictably happy ending."

# 5
def system_prompt_template(user_description, list_of_item_description):
    return f"You are an Amazon buyer. Here is your self-introduction, expressing your preferences and dislikes: '{user_description}'. \n\n Now, you are considering selecting an item from two candidates. The features of these items are:\n {list_of_item_description}.\n\n Please select the item that aligns best with your preferences and explain your choice while rejecting the other. \n Follow these steps:\n 1. Extract your preferences and dislikes from your self-introduction. \n 2. Evaluate the two items based on your preferences and how they relate to the item features.\n 3. Explain your choice, detailing the relationship between your preferences/dislikes and the item features. \n\n Important notes:\n 1. **Output Format:** 'Choice: [Title of the selected item] \\n Explanation: [Rationale behind your choice and reasons for rejecting the other item]'. \n 2. Do not fabricate your preferences! If your self-introduction lacks relevant details, use common knowledge to guide your decision, such as item popularity. \n 3. Select one candidate, not both. \n 4. Your explanation should be specific; general preferences like genre are insufficient. Focus on the item's finer attributes and be concise! \n 5. Base your explanation on facts. If your self-introduction doesn't specify preferences, you cannot claim your decision was influenced by them."

def system_prompt_crossdomain(cross_domain_preference, private_domain_description, main_kind):
    return f"As an Amazon buyer, here is your previous self-introduction: {cross_domain_preference}. Now your preferences across various product domains are outlined as follows: {private_domain_description}. Analyze these preferences across different domains to deduce your likely inclinations within {main_kind} domain. **Output format: 'My deduced preference: [description]' and keep it under 180 words. Important notes: 1. Concentrate on the preferences within the {main_kind} domain that may align with your preferences in other product domains. 2. Directly present your analyzed product preferences in the {main_kind} domain without referencing other product domains."
# 6
def system_prompt_template_evaluation_basic(user_description, candidate_num, example_list_of_item_description):
    return f"I am an Amazon buyer. My preferences: '{user_description}'.\n\nRank these {candidate_num} items based on my preferences:\n{example_list_of_item_description}\n\nOutput format:\nRank:\n1. item title\n2. item title\n3. item title\n...\n{candidate_num}. item title\n\nOnly output the ranking list."

def system_prompt_template_evaluation_basic_g(user_description, candidate_num, example_list_of_item_description, group_Mem_txt):
    return f"I am an Amazon buyer. Here is my self-introduction, which includes my preferences and dislikes:\n\n '{user_description}'. {group_Mem_txt} \n\n Now, I am looking for items that match my preferences from {candidate_num} candidates. The features of these items are as follows:\n {example_list_of_item_description}. \n\n Please rearrange these items based on my preferences and dislikes by following these steps:\n 1. Analyze my preferences and dislikes from my self-introduction. \n 2. Compare the candidate items according to my preferences, then make a recommendation. \n 3. Consider the recent interactions and choices of users with similar tastes, as their preferences may influence mine. \n 4. **Output Format: Your ranking result must follow this format:** 'Rank: {{1. item title \\n 2. item title ...}}.' \n Note: List each item title on a new line."

def system_prompt_template_evaluation_sequential(user_description, historical_interactions, candidate_num, example_list_of_item_description):
    return f"I am an Amazon buyer. Here is my self-introduction, exhibiting my preferences and dislikes: '{user_description}'. Additionally, here is my purchasing history: \n {historical_interactions}. \n\n Now, I want to find items that match my preferences from {candidate_num} candidates. The features of these candidate items are as follows:\n {example_list_of_item_description}. \n\n Please rearrange these items based on my preferences and dislikes. To do this, follow these steps:\n 1. Analyze my preferences and dislikes from my self-introduction. \n 2. Compare the candidate items according to my preferences, and make a recommendation. Consider how these items relate to my previous purchases. \n 3. Please output your recommendation in the following format: 'Rank: {{1. item title \\n 2. item title ...}}.' \n Note that the rank list should be separated by line breaks."

def system_prompt_template_evaluation_sequential_g(user_description, historical_interactions, candidate_num, example_list_of_item_description, group_Mem_txt):
    return f"I am an Amazon buyer. Here is my self-introduction, exhibiting my preferences and dislikes: '{user_description}'. Additionally, here is my purchasing history: \n {historical_interactions}. {group_Mem_txt} \n\n Now, I want to find items that match my preferences from {candidate_num} candidates. The features of these candidate items are as follows:\n {example_list_of_item_description}. \n\n Please rearrange these items based on my preferences and dislikes. To do this, follow these steps:\n 1. Analyze my preferences and dislikes from my self-introduction. \n 2. Compare the candidate items according to my preferences, and make a recommendation. Consider how these items relate to my previous purchases. \n 3. Please output your recommendation in the following format: 'Rank: {{1. item title \\n 2. item title ...}}.' \n Note that the rank list should be separated by line breaks."

def system_prompt_template_evaluation_retrieval(user_past_description, user_description, candidate_num, example_list_of_item_description):
    return f"I am an Amazon buyer. Here is my previous self-introduction, showing my past preferences and dislikes: '{user_past_description}'.\n\n Recently, I encountered some items and updated my self-introduction: '{user_description}'. \n\n Now, I want to find items that match my preferences from {candidate_num} candidates. The features of these items are as follows:\n {example_list_of_item_description}. \n\n Please rearrange these items based on my preferences and dislikes. To do this, follow these steps:\n 1. Analyze my past preferences from my previous self-introduction. \n 2. Analyze my current preferences from my updated self-introduction. \n 3. Compare the candidate items and assess their relationships to my preferences and dislikes. Rearrange them based on your analysis. \n 4. Generate your output in the following format: 'Rank: {{1. item title \\n 2. item title ...}}.' \n Note that the rank list should be separated by line breaks. \n\n Important note:\n When recommending items, prioritize my current preferences. However, my past preferences are also valuable. If unsure, refer to my past preferences and dislikes."

def system_prompt_template_evaluation_retrieval_g(user_past_description, user_description, candidate_num, example_list_of_item_description, group_Mem_txt):
    return f"I am an Amazon buyer. Here is my previous self-introduction, showing my past preferences and dislikes: '{user_past_description}'.\n\n Recently, I encountered some items and updated my self-introduction: '{user_description}'. \n\n {group_Mem_txt} Now, I want to find items that match my preferences from {candidate_num} candidates. The features of these items are as follows:\n {example_list_of_item_description}. \n\n Please rearrange these items based on my preferences and dislikes. To do this, follow these steps:\n 1. Analyze my past preferences from my previous self-introduction. \n 2. Analyze my current preferences from my updated self-introduction. \n 3. Compare the candidate items and assess their relationships to my preferences and dislikes. Rearrange them based on your analysis. \n 4. Generate your output in the following format: 'Rank: {{1. item title \\n 2. item title ...}}.' \n Note that the rank list should be separated by line breaks. \n\n Important note:\n When recommending items, prioritize my current preferences. However, my past preferences are also valuable. If unsure, refer to my past preferences and dislikes."

def get_user_tag_prompt(user_description):
    return f"Please analyze the following self-description of a user and extract multiple interest tags based on their preferences and interests. \n\nSelf-description:{user_description} \n\nOutput the tags in valid JSON format without any extra Markdown or code block indicators. Output format example: \n\n {{\"interest_tags\":[tag1, tag2, ...]}}"


def get_call_llm_for_summary(tag_list):
    return f"Please analyze the following self-description of a user and extract multiple interest tags specifically highlighting their preferences and interests related to the distinctive styles and features of products. \n\nTag list: {tag_list}\n\nInstructions:\n1. The output must be a single phrase. Do not include sentences, lists, or other formats.\n2. The phrase should be as concise and accurate as possible in summarizing all the tags.\n3. There is no need to explain or provide additional information. Just give the summary phrase."

def groupMem_summary(group_Mem_txt):
    return f"Please summarize the following group memories and output the results. Text to be summarized: {group_Mem_txt} \nRequirements:\n1.Summarize the text to ensure concise. 2. Ensure that the rewrite maintains the core points of the group memories.\n3. Highlight the recent preferences of users in different interest groups to ensure the summary is representative.\n4. Follow the output format example: 'Users who have similar preferences to me in ... recently ...,' and ensure that the rewritten results are uniformly formatted.\n5. If necessary, enhance the content with your own understanding and analysis to enrich the summary."

def baseline_llmrank(user_his_text, recent_item, recall_budget, candidate_text_order):
    return f"I have purchased items like: {user_his_text}. Now, take a look at these {recall_budget} products: {candidate_text_order}. Could you provide a ranking for these items based on the history? Please format it as 'Rank: {{1. item title \\n 2. item title ... \\n 10. item title}}.' Remember: 1. use only the information given and avoid making any assumptions about the products, 2. just provide the final output, 3. ensure the rank list is clearly separated by line breaks."

# ============= 消融实验专用 Prompts =============

# Auto模式：直接给定正负样本标签（无决策环节）
def user_prompt_auto(user_description, list_of_item_description, pos_item_title, neg_item_title):
    """Auto模式：用户智能体直接根据标准答案更新记忆（无需判断对错）"""
    return f"You are an Amazon buyer.\n Here is your previous self-introduction, exhibiting your past preferences and dislikes:\n '{user_description}'.\n\n Recently, you encountered two items. The features of these items are:\n {list_of_item_description}.\n\n After encountering these items, you found that you really like '{pos_item_title}' and dislike '{neg_item_title}'.\n Your task now is to update your self-introduction based on these direct experiences. \n Follow these steps: \n 1. Identify new preferences from '{pos_item_title}' that you like. \n 2. Identify new dislikes from '{neg_item_title}' that you dislike. \n 3. Summarize your past preferences and dislikes from your previous self-introduction, combining them with new insights and removing conflicting parts.\n 4. Update your self-introduction, starting with new preferences, then summarizing past ones, followed by dislikes. \n\n Important notes:\n 1. Your output format should be: 'My updated self-introduction: [Your updated self-introduction here].' \n 2. Keep it under 150 words.  \n 3. Be concise and clear. \n 4. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 5. Your self-introduction should be specific and personalized; avoid generic preferences."

# ============= History-based LLM Baseline Prompt =============

def history_based_ranking_prompt(historical_items_text, candidate_num, candidate_items_text):
    """基于历史交互的排序prompt（严格baseline，仅基于标题文本相似性）"""
    return f"""Task: Rank items based on title text similarity only.

Previously interacted items (titles only):
{historical_items_text}

Candidate items to rank (titles only):
{candidate_items_text}

Instructions:
- Rank the {candidate_num} candidate items based ONLY on how similar their titles are to the previously interacted item titles
- Do NOT use any external knowledge about these items (genre, artist, content, etc.)
- Do NOT infer user preferences or item characteristics
- Only compare the TEXT of the titles: look for similar words, phrases, or patterns
- Items with titles more textually similar to the history should rank higher
- This is a pure text matching task

Output format:
Rank:
1. [item title]
2. [item title]
3. [item title]
...
{candidate_num}. [item title]"""


# ============= 长记忆评估 Prompt =============

def system_prompt_template_long_memory_evaluation(user_description, round_idx, candidate_num, example_list_of_item_description):
    return (
        f"I am an Amazon buyer. Here is my self-introduction captured after round {round_idx} of my interaction history: "
        f"'{user_description}'.\n\n"
        f"Rank these {candidate_num} items based on my current preferences:\n"
        f"{example_list_of_item_description}\n\n"
        "Output format:\n"
        "Rank:\n"
        "1. item title\n"
        "2. item title\n"
        "3. item title\n"
        "...\n"
        f"{candidate_num}. item title\n\n"
        "Only output the ranking list."
    )

LONG_MEMORY_PROMPTS = {
    "long_basic": system_prompt_template_long_memory_evaluation,
}



# 添加
# ============= 属性级别监督 Prompts =============

# 属性维度定义（建议放在 config 中）
# 针对 CDs 推荐数据集优化的维度
ATTRIBUTE_DIMENSIONS = [
    "genre",           # 流派 (Rock, Jazz, Classical等)
    "artist_style",    # 艺术家风格 (Vocal, Instrumental, Experimental)
    "audio_quality",   # 音质 (Remastered, Live Recording, Studio)
    "release_era",     # 发行年代 (80s Nostalgia, Modern Release)
    "rarity",          # 稀缺性 (Limited Edition, Collector's Item, Second-hand)
    "price",           # 价格
    "label",           # 出版厂牌 (如 Sony, Blue Note - 某些用户有厂牌忠诚度)
    "mood"             # 情感/氛围 (Relaxing, Energetic, Melancholic)
]

def attribute_analysis_prompt_correct(user_description, pos_item_title, neg_item_title,
                                     pos_item_desc, neg_item_desc, system_reason):
    """当模型选对时：分析为什么选对了，强化这些属性"""
    attr_list = ", ".join(ATTRIBUTE_DIMENSIONS)
    return f"""User preferences: '{user_description}'.
The model CORRECTLY chose '{pos_item_title}' over '{neg_item_title}'.
Reasoning provided: '{system_reason}'.

Analyze which attributes confirmed this correct choice. For each relevant dimension ({attr_list}):
1. Identify the advantage of '{pos_item_title}'
2. Polarity: positive (aligns with user)
3. Importance: 1-5

Output format: [attribute]: {pos_item_title} | positive | [score]"""

def attribute_analysis_prompt_incorrect(user_description, pos_item_title, neg_item_title,
                                       pos_item_desc, neg_item_desc, system_reason):
    """当模型选错时：分析为什么被 neg 误导了，哪些属性是负面的"""
    attr_list = ", ".join(ATTRIBUTE_DIMENSIONS)
    return f"""User preferences: '{user_description}'.
The model INCORRECTLY chose '{neg_item_title}' instead of '{pos_item_title}'.
Flawed reasoning: '{system_reason}'.

Identify the attributes that led to this error. For relevant dimensions ({attr_list}):
1. Identify the misleading trait in '{neg_item_title}' or the missed trait in '{pos_item_title}'
2. Polarity: negative (conflicts with true user preference) or positive (what we should have liked)
3. Importance: 1-5

Output format: [attribute]: [item_name] | [positive/negative] | [score]"""


def user_prompt_template_with_attr(list_of_item_description, pos_item_title, neg_item_title, system_reason, attribute_dimensions):
    return f"""Recently, you considered choosing one item from two candidates. The features of these items are:
 {list_of_item_description}.

 After comparing based on your preferences, you chose '{neg_item_title}' and rejected the other. Your explanation was:
 '{system_reason}'. 

 However, after encountering these items, you realized you prefer '{pos_item_title}' and don't like '{neg_item_title}'.
 This indicates an incorrect choice, and your previous judgment about your preferences was mistaken. Your task now is to extract the attribute-level rationale for this correction and update your self-introduction. 

 Follow these steps: 
 1. Attribute Analysis (Silver Rationale): From these dimensions ({attribute_dimensions}), identify 1 to 3 key attributes that caused this misconception. For each, specify which item has the trait, its polarity (positive/negative to your true preference), and its importance score (1-5).
 2. Analyze misconceptions in your previous judgment and correct them based on step 1.
 3. Identify new preferences from '{pos_item_title}' and dislikes from '{neg_item_title}'. 
 4. Summarize your past preferences, merging them with new insights and removing conflicting parts.
 5. Update your self-introduction, starting with new preferences, then summarizing past ones, followed by dislikes. 

 Important notes:
 1. Your output format MUST strictly follow this structure:
 Attribute Rationale:
 - [attribute_1]: [item_name] | [positive/negative] | [score 1-5]
 - [attribute_2]: [item_name] | [positive/negative] | [score 1-5]
 My updated self-introduction: [Your updated self-introduction here]. 
 2. Keep the self-introduction under 150 words.  
 3. Be concise and clear. 
 4. Describe only the features of items you prefer or dislike, without mentioning your thought process in the self-introduction. 
 5. Your self-introduction should be specific and personalized; avoid generic preferences."""


def user_prompt_template_true_with_attr(list_of_item_description, pos_item_title, neg_item_title, system_reason, attribute_dimensions):
    return f"""Recently, you considered choosing one item from two candidates. The features of these items are:
 {list_of_item_description}.

 After comparing based on your preferences, you selected '{pos_item_title}' and rejected the other. Your explanation was:
 '{system_reason}'. 

 After encountering these items, you found that you really like '{pos_item_title}' and dislike '{neg_item_title}'.
 This indicates you made a correct choice, and your judgment about your preferences was accurate. 
 Your task now is to extract the attribute-level rationale confirming your choice and update your self-introduction. 

 Please follow these steps: 
 1. Attribute Analysis (Silver Rationale): From these dimensions ({attribute_dimensions}), identify 1 to 3 key attributes that drove this successful match. For each, specify which item has the trait, its polarity (positive/negative to your preference), and its importance score (1-5).
 2. Analyze your judgment about your preferences and dislikes from your explanation based on step 1.
 3. Identify new preferences based on '{pos_item_title}' and dislikes based on '{neg_item_title}'. 
 4. Summarize your past preferences and dislikes from your previous self-introduction, combining them with new insights while removing conflicting parts.
 5. Update your self-introduction, starting with your new preferences, then summarizing past ones, followed by your dislikes. 

 Important notes:
 1. Your output format MUST strictly follow this structure:
 Attribute Rationale:
 - [attribute_1]: [item_name] | [positive/negative] | [score 1-5]
 - [attribute_2]: [item_name] | [positive/negative] | [score 1-5]
 My updated self-introduction: [Your updated self-introduction here]. 
 2. Keep the self-introduction under 150 words. 
 3. Be concise and clear. 
 4. Describe only the features of items you prefer or dislike, without mentioning your thought process in the self-introduction. 
 5. Your self-introduction should be specific and personalized."""


def item_prompt_template_with_attr(user_description, list_of_item_description, pos_item_title, neg_item_title, system_reason, attribute_dimensions):
    return f"""User self-introduction, showing preferences and dislikes: '{user_description}'.
 Recently, the user browsed a shopping site and considered two items:
 {list_of_item_description}.

 He chose '{neg_item_title}' and rejected the other, explaining: '{system_reason}'. 

 However, he prefers '{pos_item_title}' instead, indicating an unsuitable choice due to misleading descriptions. He likes '{pos_item_title}' for its features and dislikes '{neg_item_title}' for undesirable traits. Your task is to extract the attribute-level rationale for this gap and update the item descriptions. 

 Follow these steps:
 1. Attribute Analysis (Silver Rationale): From these dimensions ({attribute_dimensions}), identify 1 to 3 key attributes where the original descriptions misled the user. For each, specify the item, its true polarity (positive/negative to the user), and its importance score (1-5).
 2. Analyze features that led to the poor choice and modify them based on step 1. 
 3. Examine user preferences and dislikes; explore new features of the preferred item aligning with preferences and opposing dislikes, and do the same for the disliked item, highlighting differences. 
 4. Incorporate new features into the previous descriptions, preserving valuable content while being concise.

 Important notes: 
 1. Your output format MUST strictly follow this structure:
 Attribute Rationale:
 - [attribute_1]: [item_name] | [positive/negative] | [score 1-5]
 - [attribute_2]: [item_name] | [positive/negative] | [score 1-5]
 The updated description of the first item is: [updated description]. 
 The updated description of the second item is: [updated description].
 2. Each updated description cannot exceed 50 words; be concise and clear. 
 3. In your descriptions, refer to user preferences collectively, avoiding specific individual references, e.g., 'the user with ... preferences/dislikes'.
 4. The updated description should not contradict the item's inherent characteristics. 
 5. The updated description should highlight distinguishing features that differentiate this item from others."""


def item_prompt_template_true_with_attr(user_description, list_of_item_description, pos_item_title, neg_item_title, attribute_dimensions):
    return f"""User self-description, showcasing preferences and dislikes: '{user_description}'.
 Recently, the user browsed a shopping site and considered two items:
 {list_of_item_description}.

 The user chose '{pos_item_title}' for its features and rejected '{neg_item_title}' for undesirable traits. Your task is to extract the attribute-level rationale confirming this choice and update the descriptions of these items. 

 Follow these steps:
 1. Attribute Analysis (Silver Rationale): From these dimensions ({attribute_dimensions}), identify 1 to 3 key attributes that drove this successful choice. For each, specify the item, its polarity (positive/negative to the user), and its importance score (1-5).
 2. Analyze the user's preferences and dislikes from the self-description based on step 1. 
 3. Explore the chosen item's features that align with preferences and oppose dislikes, and examine the rejected item's features that align with dislikes and oppose preferences. Highlight the differences thoroughly. 
 4. Incorporate new features into the previous descriptions, preserving key information while being concise.

 Important notes: 
 1. Your output format MUST strictly follow this structure:
 Attribute Rationale:
 - [attribute_1]: [item_name] | [positive/negative] | [score 1-5]
 - [attribute_2]: [item_name] | [positive/negative] | [score 1-5]
 The updated description of the first item is: [updated description]. 
 The updated description of the second item is: [updated description].
 2. Each updated description cannot exceed 50 words; be concise and clear! 
 3. In your updated descriptions, refer to preferences collectively.
 4. New features should reflect user preferences, and the updated descriptions must not contradict the inherent characteristics of the items."""

