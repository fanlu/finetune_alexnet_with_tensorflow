class GBaseClass:
    def __init__(self, name, value, type):
        self.name = name
        self.type = type
        self.value = value

    def partialGradient(self, partial):
        pass

    def expression(self):
        pass


G_STATIC_VARIABLE = {}


class GConstant(GBaseClass):
    def __init__(self, value):
        global G_STATIC_VARIABLE
        try:
            G_STATIC_VARIABLE['counter'] += 1
        except:
            G_STATIC_VARIABLE.setdefault('counter', 0)
        self.value = value
        self.name = "CONSTANT_" + str(G_STATIC_VARIABLE['counter'])
        self.type = "CONSTANT"

    def partialGradient(self, partial):
        return GConstant(0)

    def expression(self):
        return str(self.value)


class GVariable(GBaseClass):
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        self.type = "VARIABLE"

    def partialGradient(self, partial):
        if partial.name == self.name:
            return GConstant(1)
        return GConstant(0)

    def expression(self):
        return str(self.name)


def GOperationWrapper(left, right, operType):
    return GOperation(left, right, operType)


class GOperation(GBaseClass):
    def __init__(self, a, b, operType):
        self.operatorType = operType;
        self.left = a
        self.right = b

    def partialGradient(self, partial):
        # partial must be a variable
        if partial.type != "VARIABLE":
            return None
        if self.operatorType == "plus" or self.operatorType == "minus":
            return GOperationWrapper(self.left.partialGradient(partial), self.right.partialGradient(partial), "plus")
        if self.operatorType == "multiple":
            part1 = GOperationWrapper(self.left.partialGradient(partial), self.right, "multiple")
            part2 = GOperationWrapper(self.left, self.right.partialGradient(partial), "multiple")
            return GOperationWrapper(part1, part2, "plus")
        # pow should be g^a,a is a constant.
        if self.operatorType == "pow":
            c = GConstant(self.right.value - 1)
            part2 = GOperationWrapper(self.left, c, "pow")
            part3 = GOperationWrapper(self.right, part2, "multiple")
            return GOperationWrapper(self.left.partialGradient(partial), part3, "multiple")

    def expression(self):
        if self.operatorType == "plus":
            return self.left.expression() + "+" + self.right.expression()
        if self.operatorType == "minus":
            return self.left.expression() + "-" + self.right.expression()
        if self.operatorType == "multiple":
            return "(" + self.left.expression() + ")*(" + self.right.expression() + ")"
        # pow should be x^a,a is a constant.
        if self.operatorType == "pow":
            return "(" + self.left.expression() + ")^(" + self.right.expression() + ")"


# case 1
X = GVariable("X")
y = GVariable("y")
B = GVariable("B")
xb = GOperationWrapper(X, B, 'multiple')
xb_y = GOperationWrapper(xb, y, 'minus')
f = GOperationWrapper(xb_y, GConstant(2), 'pow')
print(f.expression())
print(f.partialGradient(B).expression())
