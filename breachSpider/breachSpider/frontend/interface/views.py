from django.shortcuts import render, redirect
from django.http import HttpResponse

import sys
sys.path.append("..")
sys.path.append("..")
import database.database as db

def index (request):
    if request.method == 'GET':
        results = db.db_grab_unconfirmed()
        context = {}
        i = 0
        while i<5 and results.alive:
            item = results.next()
            cv = item["ConfidenceValue"]
            cv *= 100
            cvs = str(cv)+"%"
            subitem = {
                "id": item["_id"],
                "ScrapyRedirectURL": item["ScrapyRedirectURL"],
                "ConfidenceValue": cvs
            }
            context[i] = subitem
            i+=1
        results.close()
        return render(request, 'interface/index.html', {'context': context})
    if request.method == 'POST':
        req = request.POST
        for id in req.keys():
            if id == 'csrfmiddlewaretoken':
                continue
            elif req[id] == 'unchanged':
                continue
            elif req[id] == 'breach':
                db.db_change_confirmed(id, True)
            elif req[id] == 'nonbreach':
                db.db_change_breach(id, False)
                db.db_change_confirmed(id, True)
        return redirect("/")