from scheduler import Scheduler

schedule = Scheduler(2, 100, [])

prompts = [
    "In the future, artificial intelligence will",
    "Scientists have discovered a new way to",
    "On a distant planet, aliens are",
    "One of the greatest inventions in human history is",
    "In a peaceful village, the residents daily",
    "In a mysterious forest, explorers found",
    "In a bustling city, people are",
    "In an ancient castle, there is a hidden",
    "In a future society, robots will",
    "In a magical world, wizards can",
    "On a remote island, the expedition team discovered",
    "In a peaceful country, people enjoy",
    "In a challenging environment, warriors must",
    "In a technologically advanced future, people can",
    "In a place full of wonders, children can",
    "In an ancient legend, heroes must",
    "On an adventurous journey, explorers will",
    "In a mysterious cave, archaeologists found",
    "In a future city, transportation will",
    "In a world full of hope, people can"
]

res = schedule.process(prompts)
for index, string in enumerate(res):
    print(index, "-" * 20)
    print(string)