import pandas as pd
import re
import nltk
from nltk import pos_tag, RegexpParser, word_tokenize
import argparse
import email

# Helper function to extract content from email
def get_text_from_email(content):
    '''Extracts text from the email content.'''
    return content if content else ""

# Helper function to split multiple email addresses
def split_email_addresses(line):
    '''Splits email addresses into a frozenset.'''
    return frozenset(map(lambda x: x.strip(), line.split(','))) if line else None

# Helper function to check for scheduled activity in a sentence
def is_scheduled_activity(phrase):
    '''Identifies if the sentence contains a scheduled activity.'''
    unwanted_chars = [',', ';', ':', '*', '-', '+', '_', '#', '!', '=', '?', '(', ')', '[', ']', '{', '}', '%', '$']
    translation_table = str.maketrans(dict.fromkeys(unwanted_chars, " "))
    phrase = phrase.translate(translation_table).strip()
    phrase = re.sub(' +', ' ', phrase)
    words = word_tokenize(phrase)
    pos_tags = pos_tag(words)
    tags = [tag for _, tag in pos_tags]

    # Define patterns to identify scheduled activity-related phrases
    try:
        if tags.index('VBZ') and tags.index('VBG'):
            return True
    except:
        pass
    try:
        if tags.index('MD') and tags.index('VB'):
            return True
    except:
        pass
    try:
        if tags.index('VBP'):
            return True
    except:
        pass

    pattern_1 = r"(?:\w+\s+)+(?:VBZ\s+VBG|MD\s+VB)\s+(?:\w+\s+)*(?:appointment|meet|meeting|call|reunion|event)+\b"
    pattern_2 = r"(?:\w+\s+)+VBP\s+(?:\w+\s+)*(?:appointment|meet|meeting|call|reunion|event)+\b"
    combined_tag = " ".join(tags)

    return bool(re.match(pattern_1, combined_tag) or re.match(pattern_2, combined_tag))

# Helper function to check if a sentence is a reminder
def is_reminder(sentence):
    '''Checks if the sentence is a reminder.'''
    tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(tokens)

    if tagged_sent[-1][0] != "?":
        if tagged_sent[0][1] in ["VB", "MD"]:
            return True
        elif tagged_sent[0][1] == "NN" and tagged_sent[1][1] == "IN":
            tokens.remove(tokens[1])
            sentence = " ".join(tokens)
            if pos_tag(word_tokenize(sentence))[0][1] == "VB":
                return True
    else:
        if "please" in [w[0].lower() for w in tagged_sent] and tagged_sent[0][1] in ["VB", "MD"]:
            return True
    return False

# Chunking helper function for grammatical phrases
def get_chunks(tagged_sent):
    '''Creates grammatical phrases based on POS tags.'''
    chunkgram = r"""
        VB-Phrase: {<DT><,>*<VB>} {<RB><VB>} {<UH><,>*<VB>} {<UH><,><VBP>} {<PRP><VB>} {<NN.?>+<,>*<VB>}
        NN-Prep-Phrase: {<VB><IN>?<NN.?>*}
        VB-Prep-Phrase: {<VB><IN><NN.*>+}
        Q-Tag: {<,><MD><RB>*<PRP><.>*}
    """
    chunkparser = RegexpParser(chunkgram)
    return chunkparser.parse(tagged_sent)

# Text classification function
def classify_text(text):
    '''Classifies text as "reminder", "scheduled activity", or "other".'''
    if is_reminder(text):
        return "reminder"
    if is_scheduled_activity(text):
        return "scheduled activity"
    return "other"

# Function to annotate email data
def annotate_data(analysis_df):
    '''Annotates each sentence in the emails.'''
    annotated_df = pd.DataFrame(columns=['content', 'label'])
    
    for text in analysis_df['content']:
        sentences = nltk.sent_tokenize(text)
        cleaned_sentences = [i.lower().strip().replace('\n', ' ') for i in sentences if i.strip()]
        
        for sentence in cleaned_sentences:
            new_row = {'content': sentence, 'label': classify_text(sentence)}
            annotated_df = pd.concat([annotated_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return annotated_df

# Function to balance the annotated dataset
def balance_dataset(annotated_df):
    '''Balances the dataset by sampling an equal number of examples from each class.'''
    class_counts = annotated_df['label'].value_counts()
    min_count = class_counts.min()

    balanced_df = pd.concat([
        annotated_df[annotated_df['label'] == label].sample(min_count, random_state=42)
        for label in class_counts.index
    ])

    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    

def parse_email_fields(emails_df):
    '''Parses email fields from the 'message' column and extracts content.'''
    # Parse the emails into a list of email objects
    messages = list(map(email.message_from_string, emails_df['message']))
    
    # Drop the original 'message' column
    emails_df.drop('message', axis=1, inplace=True)

    # Get fields from parsed email objects (e.g., From, To, Date, etc.)
    keys = messages[0].keys()
    for key in keys:
        emails_df[key] = [doc[key] for doc in messages]

    # Parse the content from email objects
    emails_df['content'] = list(map(get_text_from_email, messages))

    # Split multiple email addresses
    emails_df['From'] = emails_df['From'].map(split_email_addresses)
    emails_df['To'] = emails_df['To'].map(split_email_addresses)

    # Extract the root of 'file' as 'user'
    emails_df['user'] = emails_df['file'].map(lambda x: x.split('/')[0])
    del messages

    return emails_df


if __name__ == "__main__":
    # Argument parser for input file
    parser = argparse.ArgumentParser(description='Email Data Annotation Script')
    parser.add_argument('--input', type=str, required=True, help='Path to the input email CSV file')
    args = parser.parse_args()

    # Load the dataset
    emails_df = pd.read_csv(args.input, on_bad_lines='skip')

    # Parse the email fields and content
    emails_df = parse_email_fields(emails_df)

    # Clean up the DataFrame
    emails_df = emails_df[['From', 'To', 'Date', 'content']].dropna().copy()
    analysis_df = emails_df.loc[emails_df['To'].map(len) == 1]

    # Annotate and balance data
    annotated_df = annotate_data(analysis_df)
    balanced_df = balance_dataset(annotated_df)

    # Save results
   # analysis_df.to_excel('output_clean.xlsx', index=False)
   # annotated_df.to_excel('annotated_data.xlsx', index=False)
    balanced_df.to_excel('data/balanced_annotated_data.xlsx', index=False)

    print("Data annotation and balancing completed successfully!")
