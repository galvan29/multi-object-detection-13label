# How to commit
- Code commit need to be simple to understand
- DONT! use capital letters; use them only to say TODO, TO FIX ecc
    - src: added you  ------------- YES
    - src: Added You  ------------- NO
- Here a two examples
## NOT TO DO !!
- main.py file
```python
for i in range(1, 10):
    print("Hello World!")
```
- on the console
```bash
git add . 
git commit -m "Good code very nice"
git push
```
- note that "Good code very nice" is not a good descriptio on the previous python code

## WHAT TO DO :)
- main.py file
```python
for i in range(1, 10):
    print("Hello World!")
```
- on the console
```bash
git add . 
git commit -m "src: added loop to test print"
git push
```
- note that you start with "src:" which is the foldel where main.py is stored, followerd by a description of what you did