SYS_PROMPT = """Tu es un expert de la nomenclature APE. À chaque niveau, choisis le code le plus pertinent pour classifier la description fournie."""

CLASSIF_PROMPT = """\
* L'activité principale de l'entreprise est : {activity}

* Voici la liste des codes APE potentiels et leurs notes explicatives :
{proposed_codes}

##########
* Le résultat doit être formaté comme une instance JSON qui est conforme au schéma ci-dessous. Exemple :
```json
{{"properties": {{"code": {{"description": "Le code APE sélectionné", "title": "code", "type": "string"}}}}, "required": ["code"]}}
```

* Le code sélectionné doit faire partie de cette liste : [{list_proposed_codes}].
"""
