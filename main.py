import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

fields = {
    # Basic demographic variables
    'V161342': 'gender',  # 1:'male',2:'female'
    'V161310x': 'race',  # 1:'white',2:'black',3:'asian',5:'hispanic'
    'V161267': 'age',  # age in years
    'V161270': 'education',  # 1-9 high school; 10-12 some college; 13 bachelors; 14-16 advanced degree
    'V161244': 'church_goer',  # 1:'yes',2:'no'
    'V162125x': 'patriotism',
    # how does seeing the flag make you feel? 1:"extremely good", 2:"moderately good", 3:"a little good", 4:"neither
    # good nor bad", 5:"a little bad", 6:"moderately bad", 7:"extremely bad"
    'V161112': 'health_insurance',  # Does R have health insurance? 1:yes, 2:no

    # Political activity and party affiliation
    'V162174': 'discuss_politics',  # 1:'yes',2:'no'
    'V162256': 'political_interest',
    # how interested in politics? 1:"very interested", 2:"somewhat interested", 3:"not very interested",
    # 4:"not at all interested"
    'V161126': 'ideology',
    # 1:"extremely liberal", 2:"liberal", 3:"slightly liberal", 4:"moderate", 5:"slightly conservative",
    # 6:"conservative", 7:"extremely conservative"
    'V161155': 'pid3',  # 1:'Democrat',2:'Republican',3:'Independent'
    'V161158x': 'pid7',
    # 1:"strong democrat", 2:"not very strong democrat", 3:"independent, but closer to the Democratic party",
    # 4:"independent", 5:"independent, but closer to the Republican party", 6:"not very strong Republican",
    # 7:"strong Republican"
    'V162031x': 'voted',  # did you vote in 2016? 0:'no',1:'yes'
    'V162062x': 'votechoice',  # 1:"Hillary Clinton", 2:"Donald Trump", 3:"Gary Johnson", 4:"Jill Stein", 5:"Other"

    # Views on select issues
    'V162255': 'obama_muslim',  # is obama a muslim? 1:"yes",2:"no"
    'V161221': 'climate_change',  # is global warming happening or not? 1:yes, 2:no
    'V161222': 'cc_caused_by_man',
    # assuming it's happening, is it caused by humans? 1:mostly human, 2: mostly natural, 3:equally by people and nature
    'V161227': 'same_sex_wedding',
    # Do you think business owners who provide wedding-related services should be allowed to refuse services to
    # same-sex couples if same-sex marriage violates their religious beliefs, or do you think business owners should
    # be required to provide services regardless of a couple’s sexual orientation? 1:should be allowed to refuse,
    # 2:should be required to provide services
    'V161139': 'econ_health',  # Current economy good or bad? 1: very good ... 5: very bad
    'V161193': 'birthright_citizenship',  # Favor or oppose ending birthright citizenship? 1: favor, 2:oppose, 3:neither
    'V161196': 'build_wall',  # Build a wall with Mexico? 1: favor, 2:oppose, 3:neither
    'V161204': 'affirmative_action',
    # Does R favor or oppose affirmative action in universities? 1: favor, 2:oppose, 3:neither
    'V161218': 'corrupt_government',  # How many in government are corrupt? 1:all, 2:most, 3:about half, 4:a few, 5:none
    'V161228': 'bathroom_policy',
    # Transgender bathroom policy? 1: have to use bathroom of gender born with; 2: be allowed to use the bathroom of
    # their identified gender
    'V161231': 'gay_marriage',
    # position on gay marriage? 1: gay couples should be allowed to marry; 2: gay couples should be allowed to form
    # civil unions, but not marry; 3:  There should be no legal recognition of a gay or lesbian couple’s relationship.
    'V161233': 'death_penalty',  # R favor oppose death penalty for those convicted of murder? 1: favor, 2: oppose

    # Religious views
    'V161241': 'rel_important',  # Is religion important part of R life? 1: important, 2:not important
    'V161242': 'rel_daily_guidance',
    # Religion provides guidance in day-to-day living? 1:some, 2:quite a bit, 3:a great deal
    'V161243': 'rel_bible_word',
    # Is Bible word of God or men?  1: The Bible is the actual word of God and is to be taken literally,
    # word for word. 2:The Bible is the word of God but not everything in it should be taken literally; 3:The Bible
    # is a book written by men and is not the word of God.
    'V161245a': 'rel_attend_church',  # Attend church more often than once a week

    # Happy?
    'V161522': 'happy_life'  # How satisfied is R with life? 1:extremely, 2:very; 3:moderately, 4:slightly, 5:not at all
}


def simplify_anes(df):
    new_df = pd.DataFrame()
    for col in fields.keys():
        new_df[fields[col]] = df[col]
    return new_df


df = pd.read_csv('./anes2016.csv', sep='|', low_memory=False)
df = simplify_anes(df)

# only look at voters who picked Trump or Clinton
tmp = df[(df['votechoice'] == 1) | (df['votechoice'] == 2)]

# split into test and train sets
train, test = train_test_split(tmp, test_size=0.2, random_state=42)

features = set(df.columns)
features.remove("votechoice")

# construct test and train datasets
X_train = train[features]
y_train = train["votechoice"]

X_test = test[features]
y_test = test["votechoice"]

